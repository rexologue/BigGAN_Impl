'''
Functions for the main loop of training different conditional image models
'''
import os
import shutil

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
    """
    Функция train выполняет один комплексный «итерационный» шаг обучения GAN:
    1) Обновляет дискриминатор D (одновременно фиксируя генератор G).
    2) Обновляет генератор G (одновременно фиксируя дискриминатор D).
    
    Параметры:
      - G, D, gan_model: модели генератора и дискриминатора, обёрнутые в GAN.
      - x, y: текущий батч реальных данных и их меток, полученные из DataLoader.
      - z_, y_: распределения для сэмплирования латентных векторов z и случайных меток gy.
      - config: словарь с гиперпараметрами эксперимента.
    
    Важно:
      - Здесь мы используем "стандартный" подход к gradient accumulation:
        * Мы НЕ делим x и y вручную на куски;
        * Вместо этого подразумевается, что DataLoader отдаёт нам
          уже удобные небольшие батчи.
        * Для накопления градиентов мы делим лосс на 'num_*_accumulations'
          и вызываем .backward() несколько раз, а шаг оптимизатора делаем
          только после заданного количества накоплений.
    """

    # Сбрасываем градиенты оптимизаторов генератора и дискриминатора,
    # чтобы начинать тренировку «с чистого листа» на этом батче
    G.optim.zero_grad()
    D.optim.zero_grad()

    # Замораживаем генератор, размораживаем дискриминатор:
    # Именно дискриминатор сейчас будет учиться
    utils.toggle_grad(D, True)
    utils.toggle_grad(G, False)
    
    ######################
    # DISCRIMINATOR STEP #
    ######################
    #
    # Делаем num_D_steps итераций обновления дискриминатора.
    # На практике часто это число равно 1, но в некоторых случаях
    # дискриминатор обучают интенсивнее, повторяя больше шагов.
    #
    # Внутри каждого шага можно делать несколько проходов (accumulations),
    # чтобы сымитировать более крупный эффективный батч или
    # просто накопить градиенты, прежде чем делать optimizer.step().
    #
    
    # Для логгинга: инициализируем накопительные переменные лоссов дискриминатора,
    # чтобы усреднить их потом и вернуть наружу.
    D_loss_real_val = 0.0
    D_loss_fake_val = 0.0
    
    # Совокупный счётчик аккумуляций (для примера, если потребуется логика)
    accum_counter = 0
    
    for step_index in range(config['train']['num_D_steps']):
      # Перед входом в цикл аккумуляции обычно обнуляем градиенты
      # (если хотим считать этот шаг дискриминатора «отдельным»).
      # Однако при «чистом» подходе к gradient accumulation мы можем
      # обнулять их лишь после optimizer.step(), — см. чуть дальше.
      D.optim.zero_grad()
      
      # Проходим через дискриминатор num_D_accumulations раз (аккумулируем градиенты)
      for accumulation_index in range(config['train']['num_D_accumulations']):
        # Сэмплируем новые случайные латентные вектора и метки
        # и делаем forward дискриминатора одновременно с реальным батчем
        D_fake, D_real = gan_model.forward_D(
          z=z_.sample(),
          gy=y_.sample(),
          x=x,  
          dy=y
        )

        # Подсчитываем компоненты лосса дискриминатора (реальное и фейковое),
        # усредняем их и делим на количество аккумуляций, чтобы получить
        # корректный вклад от каждого backward() (вместо деления в конце).
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['train']['num_D_accumulations'])

        # Делаем backward для лосса, накапливая градиенты в D.optim
        D_loss.backward()
        
        # Для финального вычисления среднего лосса дискриминатора на этом батче
        # аккумулируем D_loss_real и D_loss_fake в отдельные переменные
        # (в конце разделим или возьмём их значения по последней итерации).
        # Здесь для простоты мы просто суммируем и в конце делим на общее число
        # итераций (num_D_steps * num_D_accumulations). Но можно сразу усреднять.
        D_loss_real_val += D_loss_real.item()
        D_loss_fake_val += D_loss_fake.item()

        accum_counter += 1

      # После окончания num_D_accumulations проходов
      # делаем шаг оптимизации дискриминатора
      if config['train']['D_ortho'] > 0.0:
        # Опциально применяем ортогональную регуляризацию к весам дискриминатора
        orthogonal_regularization(D, config['train']['D_ortho'])

      D.optim.step()
      
      # Снова обнуляем градиенты перед следующим шагом дискриминатора
      # или перед переходом к генератору
      D.optim.zero_grad()

    # В конце получаем средние значения D_loss_real_val и D_loss_fake_val.
    # Чтобы соблюсти прежнюю логику, усредним их вручную на количество
    # всех «итераций внутри дискриминатора».
    total_D_iters = config['train']['num_D_steps'] * config['train']['num_D_accumulations']
    D_loss_real_val /= float(total_D_iters)
    D_loss_fake_val /= float(total_D_iters)

    # Теперь замораживаем дискриминатор и размораживаем генератор,
    # чтобы обучить сам генератор
    utils.toggle_grad(D, False)
    utils.toggle_grad(G, True)

    ##################
    # GENERATOR STEP #
    ##################
    #
    # Здесь делаем один «комплексный» шаг генератора. Как правило,
    # обходимся одним шагом (num_G_steps = 1), но внутри него можно
    # тоже накапливать градиенты (num_G_accumulations).
    #
    
    G.optim.zero_grad()

    # Создадим переменную для лога итогового лосса генератора
    final_G_loss_val = 0.0
    
    # Проходим через генератор num_G_accumulations раз,
    # аккумулируя градиенты.
    for accumulation_index in range(config['train']['num_G_accumulations']):   
      # Сэмплируем новые случайные латентные вектора и метки, делаем forward pass
      D_fake = gan_model.forward_G(
        z=z_.sample(),
        gy=y_.sample()
      )
      
      # Подсчитываем лосс генератора, деля на количество аккумуляций
      # (чтобы каждый backward вносил вклад в итоговый градиент).
      G_loss = losses.generator_loss(D_fake) / float(config['train']['num_G_accumulations'])
      G_loss.backward()
      
      # Копим значение лосса (для лога), чтобы в конце вернуть усреднённое
      final_G_loss_val += G_loss.item()

    # Опциально применяем ортогональную регуляризацию к весам генератора
    # (исключая эмбеддинги, где это не имеет смысла).
    if config['train']['G_ortho'] > 0.0:
      orthogonal_regularization(G, config['train']['G_ortho'], 
                                blacklist=[param for param in G.shared.parameters()])
      
    G.optim.step()
    
    # Усредняем финальное значение лосса по num_G_accumulations итерациям
    final_G_loss_val /= float(config['train']['num_G_accumulations'])

    # Формируем выходной словарь с лоссами для удобства логирования
    out = {
      'G_loss': float(final_G_loss_val),
      'D_loss_real': float(D_loss_real_val),
      'D_loss_fake': float(D_loss_fake_val)
    }

    # Возвращаем лосс генератора и компоненты лосса дискриминатора
    # для внешнего контроля процесса обучения (логирования/печати).
    return out


def save_and_sample(G: Generator, 
                    G_ema: Generator, 
                    D: Discriminator, 
                    fixed_z: torch.Tensor, 
                    fixed_y: torch.Tensor, 
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
    fixed_z : torch.Tensor
        Фиксированный латентный вектор `z` для визуализации.
    fixed_y : torch.Tensor
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

    exp_state_dict['saved_ckps'].append(weights_save_path)

    # Если количество сохранённых чекпоинтов превышает 6
    if len(exp_state_dict['saved_ckps']) > 6:
      # Удаляем первые два элемента (самые старые чекпоинты)
      for ckp in exp_state_dict['saved_ckps'][:2]:  # Берём первые два элемента
        # Удаляем директорию
        shutil.rmtree(ckp)
    
    # Удаляем из списка
    exp_state_dict['saved_ckps'] = exp_state_dict['saved_ckps'][2:]
    
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
      
      best_path = os.path.join(config['weights_root'], "best_perfomance")

      shutil.rmtree(best_path)
      os.mkdir(best_path)
      
      # Сохраняем веса и результат
      exp_state_dict['best_FID'] = fid
      utils.save_weigths(G, G_ema, D, best_path)

      with open(os.path.join(best_path, 'stats.txt'), 'w', encoding='utf8') as f:
        f.write(f"Model's perfomance: {fid}\nIteration: {exp_state_dict['itr']}")
    
    return fid

    