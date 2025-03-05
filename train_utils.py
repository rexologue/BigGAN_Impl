'''
Functions for the main loop of training different conditional image models
'''
import os

import torch
import torchvision

import fid
import utils
import losses
from math_functions import orthogonal_regularization

from biggan import Generator, Discriminator, GAN

def train(G: Generator, 
          D: Discriminator, 
          gan_model: GAN, 
          x: torch.Tensor, 
          y: torch.Tensor, 
          z_: utils.Distribution, 
          y_: utils.Distribution, 
          config: dict):
    
    G.optim.zero_grad()
    D.optim.zero_grad()

    # На сколько частей разбить x и y?
    x = torch.split(x, config['train']['batch_size'])
    y = torch.split(y, config['train']['batch_size'])
    counter = 0
    
    # Замораживаем генератор
    utils.toggle_grad(D, True)
    utils.toggle_grad(G, False)
      
    ######################
    # DISCRIMINATOR STEP #
    ######################

    for step_index in range(config['train']['num_D_steps']):
      D.optim.zero_grad()

      # Проходим через дискримнатор num_D_accumulations раз
      for accumulation_index in range(config['train']['num_D_accumulations']):
        # Сэмплируем новые случайные латентные вектора и метки
        z_.sample()
        y_.sample()

        # Forward pass 
        D_fake, D_real = gan_model.forward_D(
          z=z_, 
          gy=y_, 
          x=x[counter], 
          dy=y[counter]
        )
         
        # Подсчитываем компопенты лосса дискриминатора, усредняем их и делим на количетсво аккумуляций
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['train']['num_D_accumulations'])
        D_loss.backward()

        counter += 1
        
      # Опциально применяем ортогональную регуляризацию к весам дискриминатора
      if config['train']['D_ortho'] > 0.0:
        orthogonal_regularization(D, config['train']['D_ortho'])
      
      D.optim.step()
    
    # Замораживаем дискриминатор
    utils.toggle_grad(D, False)
    utils.toggle_grad(G, True)

    ##################
    # GENERATOR STEP #
    ##################
      
    # Обнуляем градиенты генератора, на случай, если они накопились за шаг дискримнатора
    G.optim.zero_grad()
    
    # Проходим через генератор num_G_accumulations раз
    for accumulation_index in range(config['train']['num_G_accumulations']):   
      # Сэмплируем новые случайные латентные вектора и метки 
      z_.sample()
      y_.sample()

      # Forward pass 
      D_fake = gan_model.forward_G(
        z=z_, 
        gy=y_
      )
      
      # Подсчитываем лосс генератора, усредняя его делением на количетсво аккумуляций
      G_loss = losses.generator_loss(D_fake) / float(config['train']['num_G_accumulations'])
      G_loss.backward()
    
    # Опциально применяем ортогональную регуляризацию к весам генератора
    if config['train']['G_ortho'] > 0.0:
      # В ходе эмпирических наблюдений было выявлено, что применение ортогональной регуляризации 
      # к эмббедингам генератора не имеет смысла. Исключаем их, добавив в blacklist.
      orthogonal_regularization(G, config['train']['G_ortho'], blacklist=[param for param in G.shared.parameters()])
      
    G.optim.step()
    
    out = {
      'G_loss': float(G_loss.item()), 
      'D_loss_real': float(D_loss_real.item()),
      'D_loss_fake': float(D_loss_fake.item())
    }
    
    # Возрарщаем лосс генератора и компоненты лосса дискриминатора
    return out


def save_and_sample(G: Generator, 
                    G_ema: Generator, 
                    D: Discriminator, 
                    fixed_z: utils.Distribution, 
                    fixed_y: utils.Distribution, 
                    exp_state_dict: dict, 
                    config: dict):
    """
    Сохраняет веса моделей и генерирует примеры изображений для визуализации.
    
    Функция выполняет:
    1. Сохранение весов генератора, дискриминатора и EMA-версии генератора.
    2. Генерацию изображений с фиксированными `z` и `y`.
    3. Создание сэмплов для всех классов.

    Параметры:
    ----------
    G : Generator
        Генераторная модель.
    G_ema : Generator
        Экспоненциально усреднённая версия генератора (EMA).
    D : Discriminator
        Дискриминаторная модель.
    fixed_z : utils.Distribution
        Фиксированный латентный вектор `z` для визуализации.
    fixed_y : utils.Distribution
        Фиксированные метки `y` для визуализации.
    exp_state_dict : dict
        Словарь состояния эксперимента (содержит текущую итерацию и метрики).
    config : dict
        Конфигурационный словарь с путями для сохранения и параметрами модели.
    """
    # Создание директории для сохранения весов
    weights_save_path = os.path.join(config['weights_root'], f"iter_{exp_state_dict['itr']}")
    os.mkdir(weights_save_path)

    # Сохранение весов моделей
    utils.save_weigths(G, G_ema, D, weights_save_path)
    
    # Генерация изображений с фиксированными `z` и `y`
    with torch.no_grad():
        fixed_Gz = G_ema(fixed_z, G_ema.shared(fixed_y))
    
    image_filename = os.path.join(config['samples_root'], f"fixed_samples_{exp_state_dict['itr']}.jpg")
    torchvision.utils.save_image(
        fixed_Gz.float().cpu(), 
        image_filename,
        nrow=int(fixed_Gz.shape[0] ** 0.5), 
        normalize=True
    )
    
    # Генерация примеров изображений для всех классов
    utils.sample_for_all_classes(
        G_ema,
        config['n_classes'],
        config['samples_root'],
        exp_state_dict['itr']
    )


def validate(G: Generator, 
             G_ema: Generator, 
             D: Discriminator, 
             exp_state_dict: dict, 
             config: dict, 
             fid_computer: fid.FID) -> float:
    """
    Проверяет качество модели, вычисляя метрику FID, и сохраняет лучшие модели.
    
    1. Вычисляет FID (Frechet Inception Distance).
    2. Если текущий FID лучше предыдущего лучшего, сохраняет модель.
    
    Параметры:
    ----------
    G : Generator
        Генераторная модель.
    G_ema : Generator
        Экспоненциально усреднённая версия генератора (EMA).
    D : Discriminator
        Дискриминаторная модель.
    exp_state_dict : dict
        Словарь состояния эксперимента (содержит текущую итерацию и метрики).
    config : dict
        Конфигурационный словарь с путями для сохранения и параметрами модели.
    fid_computer : fid.FID
        Объект для вычисления метрики FID.
    
    Возвращает:
    -----------
    float
        Вычисленное значение FID.
    """
    # Вычисляем FID
    fid = fid_computer()
    
    # Проверяем, улучшилась ли метрика
    if exp_state_dict['best_FID'] > fid:
        print(f"FID value improved on {exp_state_dict['itr']} iteration. Saving checkpoint...")
        
        best_path = os.path.join(config['weights_root'], f"best_iter_{exp_state_dict['itr']}_fid_{round(fid)}")
        os.makedirs(best_path, exist_ok=True)
        
        exp_state_dict['best_FID'] = fid
        utils.save_weigths(G, G_ema, D, best_path)
    
    return fid

    