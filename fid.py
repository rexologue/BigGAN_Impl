import os
from tqdm import tqdm 
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3, Inception_V3_Weights

from biggan.data_utils import get_loader
from biggan.math_functions import covariance_matrix, sqrt_newton_schulz


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


def accumulate_inception_activations(sample_function: Callable, 
                                     net: torch.nn.Module, 
                                     num_inception_images=50000):
    """
    Сбор активаций Inception-сети для последующего вычисления FID.
    
    Функция выполняет проход по данным и накапливает выходные активации сети.
    
    Параметры:
    ----------
    sample_function : Callable
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
            images, labels_val = sample_function()
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
    sample_function : Callable
        Функция-сэмплер, генерирующая батчи изображений.
    """
    def __init__(self, 
                 config: dict, 
                 sample_function: Callable):
        
        self.sample_function = sample_function
        self.device = config['device']
        self.net = load_inception().to(self.device)
        self.num_inception_images = config['train']['num_inception_images']

        # Пути к файлам, где будем хранить/искать статистики
        data_mu_path = os.path.join(config['inception_root'], 'data_mu.pth')
        data_sigma_path = os.path.join(config['inception_root'], 'data_sigma.pth')

        # Если статистики уже существуют, загружаем их
        if os.path.exists(data_mu_path) and os.path.exists(data_sigma_path):
            # Загружаем предвычисленные среднее и ковариационную матрицу 
            self.data_mu = torch.load(data_mu_path).to(self.device)       # shape [D]
            self.data_sigma = torch.load(data_sigma_path).to(self.device) # shape [D, D]
        else:
            # Генерируем датасет реальных изображений для вычисления статистик
            loader = get_loader(config, 
                                batch_size=config['train']['batch_size'], 
                                start_itr=0, 
                                random_use=True)
            
            print("Start calculating means and covariances for FID...")
            
            # Вместо накопления всех активаций в pool мы считаем статистики "онлайн".
            # Если датасет большой, это позволяет избежать нехватки памяти (RAM).
            self.data_mu, self.data_sigma = self._compute_dataset_stats_streaming(loader)

            print('Saving calculated means and covariances to disk...')
            # Переводим результат на CPU и сохраняем
            torch.save(self.data_mu.cpu(), data_mu_path)
            torch.save(self.data_sigma.cpu(), data_sigma_path)


    def __call__(self) -> float:
        """
        Вычисляет FID для сгенерированных изображений.
        
        1. Генерирует сэмплы с помощью переданной `sample_function`.
        2. Вычисляет среднее и ковариацию активаций генератора.
        3. Использует `calculate_frechet_distance()` для вычисления FID.
        
        Возвращает:
        -----------
        float
            Вычисленное значение FID.
        """
        # Сбор активаций для сгенерированных изображений
        # (примерно как в вашем accumulate_inception_activations)
        pool, logits, labels = accumulate_inception_activations(
            sample_function=self.sample_function,
            net=self.net,
            num_inception_images=self.num_inception_images,
            device=self.device
        )

        # Вычисляем среднее и ковариацию в PyTorch
        mu = pool.mean(dim=0)  # [D]
        sigma = covariance_matrix(pool, rowvar=False)  # [D, D]

        # Перекладываем сохранённые статистики на нужное устройство
        real_mu = self.data_mu.to(self.device).float()
        real_sigma = self.data_sigma.to(self.device).float()

        # Вычисляем Frechet Distance
        fid = calculate_frechet_distance(mu, sigma, real_mu, real_sigma)
        fid = float(fid.cpu().item())

        # Освобождаем память
        del mu, sigma, pool, logits, labels, real_mu, real_sigma
        return fid


    def _compute_dataset_stats_streaming(self, loader):
        """
        Вычисляет среднее и ковариацию активаций Inception для реальных данных в режиме стриминга,
        не сохраняя весь 'pool' в памяти.

        Параметры:
        ----------
        loader : DataLoader
            Даталоудер с реальными изображениями.

        Возвращает:
        -----------
        mean : torch.Tensor, shape [D]
            Среднее значение активаций.
        cov : torch.Tensor, shape [D, D]
            Ковариационная матрица активаций (Bessel's correction).
        """
        self.net.eval()
        sum_features = None
        sum_features_squared = None
        n_samples = 0

        # Без вычисления градиентов
        with torch.no_grad():
            for x, _ in tqdm(loader):
                x = x.to(self.device)
                pool_val, _ = self.net(x)  # pool_val shape = [B, D]

                B, D = pool_val.shape
                if sum_features is None:
                    # Инициализируем аккумуляторы
                    sum_features = torch.zeros(D, device=self.device)
                    sum_features_squared = torch.zeros(D, D, device=self.device)

                # Накопление суммы признаков
                sum_features += pool_val.sum(dim=0)  # [D]

                # Накопление суммы внешних произведений
                # pool_val.t() shape [D, B], => matmul => [D, D]
                sum_features_squared += pool_val.t().mm(pool_val)
                n_samples += B

        # Итого считаем mean и cov
        mean = sum_features / n_samples
        # Bessel's correction => делим на (n_samples - 1).
        # (S2 - S1*S1^T / N) / (N - 1)
        # где:
        #  S1 = sum_features, shape [D]
        #  S2 = sum_features_squared, shape [D, D]
        S2 = sum_features_squared
        S1_outer = torch.outer(sum_features, sum_features)  # внешнее произведение -> [D, D]
        cov = (S2 - S1_outer / n_samples) / (n_samples - 1)

        return mean, cov
    