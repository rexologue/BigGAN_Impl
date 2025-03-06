import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3, Inception_V3_Weights

import numpy as np
from tqdm import tqdm 

from data_utils import get_loader
from math_functions import covariance_matrix, sqrt_newton_schulz


class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception, self).__init__()
    self.net = net
    self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
    
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits


def calculate_frechet_distance(mu1: torch.Tensor, 
                               sigma1: torch.Tensor, 
                               mu2: torch.Tensor, 
                               sigma2: torch.Tensor, 
                               eps=1e-6):
    """
    Вычисляет расстояние Фреше между двумя многомерными нормальными распределениями.
    
    Формула:
        d^2 = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2 * sqrt(C1 * C2))
    
    Параметры:
    ----------
    mu1 : torch.Tensor
        Вектор средних значений активаций (получен для сгенерированных данных).
    sigma1 : torch.Tensor
        Ковариационная матрица активаций для сгенерированных данных.
    mu2 : torch.Tensor
        Вектор средних значений активаций для эталонного датасета.
    sigma2 : torch.Tensor
        Ковариационная матрица активаций эталонного датасета.
    eps : float, optional
        Малое число для числовой стабильности (по умолчанию `1e-6`).
    
    Возвращает:
    -----------
    torch.Tensor
        Вычисленное расстояние Фреше между двумя распределениями.
    """
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Используем метод Ньютона-Шульца для вычисления квадратного корня из произведения ковариационных матриц
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    
    # Итоговое расстояние Фреше
    out = (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean))
    
    return out


def accumulate_inception_activations(sampler: Callable, 
                                     net: torch.nn.Module, 
                                     num_inception_images=50000):
    """
    Сбор активаций Inception-сети для последующего вычисления FID.
    
    Функция выполняет проход по данным и накапливает выходные активации сети.
    
    Параметры:
    ----------
    sampler : Callable
        Функция-семплер, которая возвращает изображения и соответствующие метки.
    net : torch.nn.Module
        Inception-сеть, используемая для извлечения активаций.
    num_inception_images : int, optional
        Число изображений, для которых будут собраны активации (по умолчанию `50000`).
    
    Возвращает:
    -----------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - `pool` — матрица активаций перед последним слоем (используется в FID).
        - `logits` — выходы сети (используется в Inception Score).
        - `labels` — метки классов (если используются для Inception Accuracy).
    """
    pool, logits, labels = [], [], []

    while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
        with torch.no_grad():
            images, labels_val = sampler()
            pool_val, logits_val = net(images.float())
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]
            labels += [labels_val]

    return torch.cat(pool, 0), torch.cat(logits, 0), torch.cat(labels, 0)


def load_inception():
    """
    Загружает предобученную модель Inception-v3 и оборачивает её в `WrapInception`.
    
    Модель используется для вычисления FID.
    
    Возвращает:
    -----------
    WrapInception
        Обернутая модель Inception для извлечения активаций.
    """
    inception = inception_v3(transform_input=False, weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception_model = WrapInception(inception.eval())
    
    return inception_model


class FID:
    """
    Класс для вычисления FID (Frechet Inception Distance) между реальными и сгенерированными изображениями.
    
    Основные функции:
    ------------------
    - Загружает предобученную Inception-сеть для вычисления активаций.
    - Загружает или вычисляет среднее значение и ковариацию активаций реальных данных.
    - Вычисляет FID для сгенерированных изображений.
    
    Параметры:
    ----------
    config : dict
        Конфигурационный словарь с настройками модели и путями к файлам.
    sampler : Callable
        Функция-сэмплер, генерирующая батчи изображений.
    """
    def __init__(self, 
                 config: dict, 
                 sampler: Callable):
        self.sampler = sampler
        self.device = config['device']
        self.net = load_inception().to(self.device)
        self.num_inception_images = config['train']['num_inception_images']

        data_mu_path = os.path.join(config['inception_root'], 'data_mu.npz')
        data_sigma_path = os.path.join(config['inception_root'], 'data_sigma.npz')

        if os.path.exists(data_mu_path) and os.path.exists(data_sigma_path):
            # Загружаем предвычисленные среднее и ковариационную матрицу
            self.data_mu = np.load(data_mu_path)['arr_0']
            self.data_sigma = np.load(data_sigma_path)['arr_0']
        else:
            # Генерируем датасет реальных изображений для вычисления статистик
            loader = get_loader(config, batch_size=config['train']['batch_size'], start_itr=0, random_use=True)
            
            print("Start calculating means and covariances for FID...")
            
            pool, logits, labels = [], [], []
            for i, (x, y) in enumerate(tqdm(loader)):
                x = x.to(self.device)

                with torch.no_grad():
                    pool_val, logits_val = self.net(x)
                    pool += [np.asarray(pool_val.cpu())]
                    logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
                    labels += [np.asarray(y.cpu())]

            pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]

            # Вычисляем среднее значение и ковариацию
            self.data_mu, self.data_sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            print('Saving calculated means and covariances to disk...')
            np.savez(data_mu_path, self.data_mu)
            np.savez(data_sigma_path, self.data_sigma)


    def __call__(self) -> float:
        """
        Вычисляет FID для сгенерированных изображений.
        
        1. Генерирует сэмплы с помощью переданного `sampler`.
        2. Вычисляет среднее и ковариацию активаций генератора.
        3. Использует `calculate_frechet_distance()` для вычисления FID.
        
        Возвращает:
        -----------
        float
            Вычисленное значение FID.
        """
        pool, logits, labels = accumulate_inception_activations(self.sampler, self.net, self.num_inception_images)

        mu, sigma = torch.mean(pool, 0), covariance_matrix(pool, rowvar=False)
        fid = calculate_frechet_distance(mu, sigma, torch.tensor(self.data_mu).float().to(self.device), torch.tensor(self.data_sigma).float().to(self.device))
        fid = float(fid.cpu().numpy())

        # Освобождаем память
        del mu, sigma, pool, logits, labels

        return fid
       
                
               

               




