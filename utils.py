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


class Distribution(torch.Tensor):
  """
  Расширенный тензор для удобного сэмплирования из заданного распределения.

  Данный класс позволяет создавать тензоры, из которых можно легко 
  сэмплировать значения, используя нормальное или категориальное распределение.
  
  Основные возможности:
  ----------------------
  1. Поддержка различных распределений:
      - `normal` (нормальное распределение с параметрами `mean` и `var`).
      - `categorical` (категориальное распределение с числом классов `num_categories`).
  
  2. Простая генерация случайных значений:
      - Метод `sample()` позволяет сэмплировать значения в соответствии с 
        заданным распределением.
      
  3. Поддержка перевода на другие устройства и типы данных:
      - Метод `to()` сохраняет параметры распределения при переводе тензора.
      
  Пример использования:
  ---------------------
  .. code-block:: python
      
      dist = Distribution(10)  # Создание тензора
      dist.init_distribution('normal', mean=0.0, var=1.0)  # Инициализация
      samples = dist.sample()  # Получение выборки
  
  Параметры:
  ----------
  dist_type : str
      Тип распределения (`'normal'` или `'categorical'`).
  mean : float, optional
      Среднее значение для нормального распределения (по умолчанию 0.0).
  var : float, optional
      Дисперсия для нормального распределения (по умолчанию 1.0).
  num_categories : int, optional
      Количество классов для категориального распределения (по умолчанию 2).
  """
  def __new__(cls, *args, **kwargs):
    return super().__new__(cls, *args, **kwargs)


  def init_distribution(self, 
                        dist_type: Literal['normal', 'categorical'], 
                        **kwargs):
    """
    Инициализирует параметры распределения для тензора.
    
    Параметры:
    ----------
    dist_type : str
        Тип распределения (`'normal'` или `'categorical'`).
    **kwargs : dict
        Дополнительные параметры распределения:
        - `mean` (среднее) и `var` (дисперсия) для нормального распределения.
        - `num_categories` (количество классов) для категориального распределения.
    """
    self.dist_type = dist_type
    self.dist_kwargs = kwargs

    if self.dist_type == 'normal':
        self.mean = kwargs.get('mean', 0.0)
        self.var = kwargs.get('var', 1.0)

    elif self.dist_type == 'categorical':
        self.num_categories = kwargs.get('num_categories', 2)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


  def sample(self):
    """
    Генерирует выборку случайных значений в соответствии с заданным распределением.
    
    Возвращает:
    -----------
    torch.Tensor
        Тензор случайных значений заданного распределения.
    """
    if self.dist_type == 'normal':
        return torch.normal(self.mean, self.var, size=self.shape)
    elif self.dist_type == 'categorical':
        return torch.randint(0, self.num_categories, size=self.shape)
    else:
        raise ValueError(f"Unknown distribution type: {self.dist_type}")
    
  
  def to(self, *args, **kwargs):
    """
    Переводит тензор на другое устройство или в другой формат данных, 
    сохраняя параметры распределения.
    
    Параметры:
    ----------
    *args, **kwargs :
        Аргументы, передаваемые в стандартный метод `torch.Tensor.to()`.
    
    Возвращает:
    -----------
    Distribution
        Новый объект `Distribution`, приведённый к указанному устройству/формату.
    """
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    
    # Вызов стандартного метода .to()
    temp = super().to(*args, **kwargs)
    
    # Если распределение категориальное, конвертируем тензор в LongTensor
    if self.dist_type == 'categorical':
        temp = temp.long()
    
    new_obj.data = temp
    
    return new_obj


def prepare_z_y(batch_size: int, 
                dim_z: int, 
                num_classes: int, 
                device='cuda', 
                z_var=1.0) -> Tuple[Distribution, Distribution]:
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
  z_var : float, optional
      Дисперсия для нормального распределения `z` (по умолчанию `1.0`).

  Возвращает:
  -----------
  Tuple[Distribution, Distribution]
      Кортеж из тензоров `z_` и `y_`.
  """
  z_ = Distribution(torch.randn(batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device)   

  y_ = Distribution(torch.zeros(batch_size, requires_grad=False))
  y_.init_distribution('categorical', num_categories=num_classes)
  y_ = y_.to(device)

  return z_, y_


def sample(G: Generator, 
           z_: Distribution, 
           y_: Distribution) -> Tuple[torch.Tensor, Distribution]:
  """
  Генерирует изображения с помощью генератора `G`, используя входные `z_` и `y_`.

  Параметры:
  ----------
  G : Generator
      Генераторная модель, принимающая `z_` и `y_` в качестве входных данных.
  z_ : Distribution
      Латентный вектор сэмплированных значений.
  y_ : Distribution
      Классовые метки сэмплированных значений.
  
  Возвращает:
  -----------
  Tuple[torch.Tensor, Distribution]
      - `G_z` — сгенерированные изображения.
      - `y_` — обновленные метки классов.
  """
  with torch.no_grad():
      z_.sample()
      y_.sample()
      G_z = G(z_, G.shared(y_))
  
  return G_z, y_


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
