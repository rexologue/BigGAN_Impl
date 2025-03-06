import os
import time
import json
from typing import Tuple, Literal

import torch
import torchvision
import numpy as np

from biggan import Generator, Discriminator

def seed_rng(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


def prepare_root(config: dict) -> dict:
  """
  Создание директорий для хранения файлов, создаваемых в ходе обучения, а именно:

  - experiment_root: Если не создана заранее - создаем. Директория всего эксперимента.
  - weights_root: Директория для хранения чекпоинтов (весов) модели.
  - logs_root: Директория с логами метрик по ходу обучения.
  - samples_root: Директория для сохранения промежуточных результатов генерации.
  - data_root: Директория для сохранения файлов, связанных с данными. (Список индексов для Sampler'а)
  - inception_root: Директория для хранения файлов, связанных с расчетом FID. Обычно средние и дисперсии исходного датасета.
  """
  if not os.path.exists(config['experiment_root']):
    print(f"Making directory '{config['experiment_root']}'")
    os.mkdir(config['experiment_root'])   
          
  for root in ['weights_root', 'logs_root', 'samples_root', 'data_root', 'inception_root']:
    root_path = os.path.join(config['experiment_root'], root)
    config[root] = root_path

    if not os.path.exists(root_path):
      print(f"Making directory '{root}' in {config['experiment_root']}")
      os.mkdir(root_path)

  return config


def toggle_grad(module: torch.nn.Module, 
                set_or_disable: bool):
  """
  Функция отключения/включения градиентов для весов модели.

  Аргументы:
    - module (torch.nn.Module): Сама модель.
    - set_or_disable (bool): Флаг, показываюший, что нужно сделать с весами модели - заморозить или разморозить.
  """
  for param in module.parameters():
    param.requires_grad = set_or_disable


def count_parameters(module: torch.nn.Module) -> int:
  """
  Функция для подсчета количества параметров модели.
  """
  return sum([p.data.nelement() for p in module.parameters()])


def load_weights(G: Generator, 
                 G_ema: Generator, 
                 D: Discriminator, 
                 load_path: str):
  """
  Загрузка модели с диска.\n
  Сразу загружает веса генератора/дискриминатора и их оптимизаторов. Ожидается, что
  load_path - директория, со слеющими файлами:

  - /load_path/G.pth ------- Веса генератора
  - /load_path/G_ema.pth --- Веса EMA генератора
  - /load_path/G_opt.pth --- Веса оптимизатора генератора
  - /load_path/D.pth ------- Веса дискриминатора
  - /load_path/D_opt.pth --- Веса оптимизатора дискриминатора
  """
  G.load_state_dict(torch.load(os.path.join(load_path, 'G.pth'), weights_only=False))
  G_ema.load_state_dict(torch.load(os.path.join(load_path, 'G_ema.pth'), weights_only=False))
  G.optim.load_state_dict(torch.load(os.path.join(load_path, 'G_opt.pth'), weights_only=False))
  D.load_state_dict(torch.load(os.path.join(load_path, 'D.pth'), weights_only=False))
  D.optim.load_state_dict(torch.load(os.path.join(load_path, 'D_opt.pth'), weights_only=False))


def save_weigths(G: Generator, 
                 G_ema: Generator, 
                 D: Discriminator, 
                 save_path: str):
  """
  Сохранение модели на диск.\n
  Сохраняет веса генератора/дискриминатора и их оптимизаторов в директорию save_path
  следующим образом:

  - /save_path/G.pth ------- Веса генератора
  - /save_path/G_ema.pth --- Веса EMA генератора
  - /save_path/G_opt.pth --- Веса оптимизатора генератора
  - /save_path/D.pth ------- Веса дискриминатора
  - /save_path/D_opt.pth --- Веса оптимизатора дискриминатора
  """
  torch.save(G.state_dict(), os.path.join(save_path, 'G.pth'))
  torch.save(G_ema.state_dict(), os.path.join(save_path, 'G_ema.pth'))
  torch.save(G.optim.state_dict(), os.path.join(save_path, 'G_opt.pth'))
  torch.save(D.state_dict(), os.path.join(save_path, 'D.pth'))
  torch.save(D.optim.state_dict(), os.path.join(save_path, 'D_opt.pth'))


class MetricsLogger:
  """
  Класс для логирования метрик во время обучения или оценки модели.

  Данный класс позволяет сохранять метрики в файл в формате JSON, 
  что упрощает анализ и последующую визуализацию данных.

  Основные возможности:
  ----------------------
  1. Автоматическая запись метрик в файл:
      - Каждая запись содержит переданные метрики и временную метку.
      - Данные записываются в файл построчно в формате JSON.
  
  2. Поддержка перезаписи файла:
      - Если файл уже существует, его можно удалить при инициализации 
        (если установлен `reinitialize=True`).
  
  Применение:
  -----------
  Используется для логирования параметров модели, значений функции потерь, 
  точности классификации и других важных характеристик в процессе обучения.

  Пример использования:
  ---------------------
  .. code-block:: python
      
      logger = MetricsLogger("metrics.log", reinitialize=True)
      
      for epoch in range(10):
          loss = compute_loss()
          accuracy = compute_accuracy()
          logger.log(epoch=epoch, loss=loss, accuracy=accuracy)
  
  Параметры:
  ----------
  fname : str
      Имя файла для сохранения метрик.
  reinitialize : bool, optional
      Если True, удаляет файл перед началом новой сессии логирования.

  Атрибуты:
  ---------
  fname : str
      Имя файла, в который будут записываться метрики.
  reinitialize : bool
      Флаг, определяющий, нужно ли удалять существующий файл перед записью новых данных.
  """
  def __init__(self, fname: str, reinitialize=False):
    self.fname = fname
    self.reinitialize = reinitialize

    # Проверяем, существует ли файл. Если да, и reinitialize=True, удаляем его.
    if os.path.exists(self.fname):
        if self.reinitialize:
            print('{} exists, deleting...'.format(self.fname))
            os.remove(self.fname)

  def log(self, **kwargs):
    """
    Записывает метрики в файл в формате JSON.

    Каждая запись сохраняется как отдельная строка в файле. 
    В метрики автоматически добавляется временная метка `_stamp`.

    Параметры:
    ----------
    **kwargs : dict
        Ключевые аргументы, представляющие записываемые метрики.
        Значения None не записываются.
    """
    # Исключаем None-значения из передаваемых метрик
    record = {k: v for k, v in kwargs.items() if v is not None}
    # Добавляем временную метку
    record['_stamp'] = time.time()

    # Открываем файл в режиме добавления ('a') и записываем строку JSON
    with open(self.fname, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=True) + '\n')


class Distribution:
    """
    Класс для генерации случайных значений из заданного распределения.

    Поддерживаются:
    - `normal` (нормальное распределение, среднее `0.0`, дисперсия `0.02`).
    - `categorical` (категориальное распределение с заданным числом классов).

    Параметры:
    ----------
    dist_type : str
        Тип распределения (`'normal'` или `'categorical'`).
    shape : tuple
        Размерность тензора выборки.
    num_categories : int, optional
        Количество классов (только для `categorical`).
    device : str, optional
        Устройство для размещения тензоров (`'cuda'` или `'cpu'`, по умолчанию `'cuda'`).
    """
    def __init__(self, 
                    dist_type: Literal['normal', 'categorical'], 
                    shape: tuple, 
                    num_categories: int = None,
                    device: Literal['cuda', 'cpu'] = 'cuda'):
        self.dist_type = dist_type
        self.shape = shape
        self.device = device

        if dist_type == 'categorical':
            self.num_categories = num_categories

        if dist_type not in ['normal', 'categorical']:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def sample(self) -> torch.Tensor:
        """
        Генерирует выборку случайных значений в соответствии с заданным распределением.
        
        Возвращает:
        -----------
        torch.Tensor
            Тензор случайных значений заданного распределения.
        """
        if self.dist_type == 'normal':
            return torch.normal(0.0, 1.0, size=self.shape).to(self.device)
        
        if self.dist_type == 'categorical':
            return torch.randint(0, self.num_categories, size=self.shape).to(self.device)


def prepare_z_y(batch_size: int, 
                dim_z: int, 
                num_classes: int, 
                device='cuda') -> Tuple[Distribution, Distribution]:
    """
    Подготавливает тензоры `z` и `y` для генератора.
    
    `z` — латентный вектор, инициализируется как нормальное распределение.
    `y` — метки классов, инициализируется как категориальное распределение.

    Параметры:
    ----------
    batch_size : int
        Размер батча.
    dim_z : int
        Размерность латентного вектора `z`.
    num_classes : int
        Количество классов.
    device : str, optional
        Устройство для размещения тензоров (по умолчанию `'cuda'`).

    Возвращает:
    -----------
    Tuple[Distribution, Distribution]
        Кортеж из объектов `z_` (латентный вектор) и `y_` (классовые метки).
    """
    z_ = Distribution(
        dist_type='normal', 
        shape=(batch_size, dim_z), 
        device=device
    )

    y_ = Distribution(
        dist_type='categorical', 
        shape=(batch_size,),  
        device=device,
        num_categories=num_classes
    )

    return z_, y_


def sample(G: Generator, 
           z_: Distribution, 
           y_: Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует изображения с помощью генератора `G`, используя входные `z_` и `y_`.

    Параметры:
    ----------
    G : Generator
        Генераторная модель, принимающая `z_` и `y_` в качестве входных данных.
    z_ : Distribution
        Генератор латентного вектора сэмплированных значений.
    y_ : Distribution
        Генератор классовых меток сэмплированных значений.
    
    Возвращает:
    -----------
    Tuple[torch.Tensor, torch.Tensor]
        - `G_z` — сгенерированные изображения.
        - `y` — обновленные метки классов.
    """
    with torch.no_grad():
        z = z_.sample()
        y = y_.sample()
        G_z = G(z, G.shared(y))
    
    return G_z, y


def sample_for_all_classes(G: Generator, 
                           num_classes: int, 
                           samples_root: str, 
                           itr: int, 
                           nrow=8):
  """
  Генерирует изображения для всех классов и сохраняет их в виде сетки.

  Параметры:
  ----------
  G : Generator
      Генераторная модель.
  num_classes : int
      Количество классов для генерации.
  samples_root : str
      Путь к директории для сохранения сгенерированных изображений.
  itr : int
      Итерация, используется для именования файлов.
  nrow : int, optional
      Количество изображений в строке при сохранении (по умолчанию `8`).
  """
  path = os.path.join(samples_root, f"iter_{itr}")
  os.makedirs(path, exist_ok=True)

  y = torch.arange(num_classes, device='cuda')
  z_ = torch.randn(num_classes, G.dim_z, device='cuda')
  
  with torch.no_grad():
      o = G(z_, G.shared(y)).cpu()
  
  torchvision.utils.save_image(
      o, 
      os.path.join(path, "all_classes.jpg"),
      nrow=nrow, 
      normalize=True
  )


def get_SVs(net: torch.nn.Module, 
            prefix: str):
  """
  Получает спектральные значения (singular values) из весов модели.

  Параметры:
  ----------
  net : nn.Module
      Нейронная сеть с параметрами, содержащими `sv` в названии.
  prefix : str
      Префикс для ключей в возвращаемом словаре.
  
  Возвращает:
  -----------
  dict
      Словарь спектральных значений, где ключи имеют формат `prefix_имя_параметра`.
  """
  d = net.state_dict()
  return {('%s_%s' % (prefix, key)).replace('.', '_'): float(d[key].item())
          for key in d if 'sv' in key}
