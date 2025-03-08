import os
import multiprocessing

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
  """
  Кастомный класс Dataset для загрузки изображений и их меток.
  
  Данный класс загружает аннотации из файла `.parquet`, содержащего пути к изображениям и их метки.
  
  Основные возможности:
  ----------------------
  - Читает аннотации из `.parquet`-файла.
  - Загружает изображения с диска и применяет трансформации (если заданы).
  - Возвращает изображение и его целевой класс в виде тензора.
  
  Параметры:
  ----------
  annot_path : str
      Путь к файлу аннотаций в формате `.parquet`.
  transforms : Callable, optional
      Функция или объект трансформации для предобработки изображений (по умолчанию `None`).
  """
  def __init__(self, annot_path: str, transforms=None):
    self.annot_df = pd.read_parquet(annot_path)  # Читаем аннотации
    self.transforms = transforms  # Сохраняем трансформации


  def __len__(self):
    """
    Возвращает количество изображений в датасете.
    
    Возвращает:
    -----------
    int
        Количество записей в аннотационном файле.
    """
    return len(self.annot_df)
  

  def __getitem__(self, idx):
    """
    Загружает изображение и соответствующую метку по индексу.
    
    Параметры:
    ----------
    idx : int
        Индекс изображения в датасете.
    
    Возвращает:
    -----------
    Tuple[torch.Tensor, torch.Tensor]
        - `img` — изображение после трансформации.
        - `target` — метка класса в формате тензора.
    """
    path = self.annot_df.loc[idx, 'path']  # Путь к изображению
    target = self.annot_df.loc[idx, 'target']  # Метка класса

    img = Image.open(path).convert('RGB')  # Загружаем изображение и конвертируем в RGB

    if self.transforms is not None:
      img = self.transforms(img)  # Применяем трансформации (если заданы)

    return img, torch.tensor(target).long()
        

class Sampler(torch.utils.data.Sampler):
  """
  Кастомный Sampler для итеративной выборки батчей из датасета.
  
  Этот Sampler позволяет:
  - Сохранять последовательность выборок для воспроизводимости.
  - Продолжать обучение с определённой итерации (`start_itr`).
  - Использовать случайные или фиксированные индексы при каждой эпохе.
  
  Параметры:
  ----------
  dataset : torch.utils.data.Dataset
      Датасет, из которого берутся индексы для обучения.
  num_epochs : int
      Количество эпох обучения.
  save_path : str
      Путь для сохранения индексов эпох (если `random_use=False`).
  start_itr : int, optional
      Итерация, с которой начинается обучение (по умолчанию `0`).
  batch_size : int, optional
      Размер батча (по умолчанию `128`).
  random_use : bool, optional
      Если `True`, индексы выбираются случайно на каждой итерации и не сохраняются (по умолчанию `False`).
  """
  def __init__(self, 
               dataset: torch.utils.data.Dataset, 
               num_epochs: int, 
               save_path: str, 
               start_itr=0, 
               batch_size=128,
               random_use=False):
    
    self.dataset = dataset
    self.num_samples = len(self.dataset)
    self.num_epochs = num_epochs
    self.start_itr = start_itr
    self.batch_size = batch_size
    self.path = os.path.join(save_path, 'sampler_dict.pth')

    # Загружаем или создаем индексы эпох
    if os.path.exists(self.path) and not random_use:
      self.epochs_itrs = torch.load(self.path, weights_only=False)
    else:
      self.epochs_itrs = [torch.randperm(self.num_samples) for _ in range(self.num_epochs)]
      if not random_use:
        torch.save(self.epochs_itrs, self.path)

    # Вычисляем текущую эпоху и итерацию
    current_epoch = self.start_itr // self.num_samples 
    if current_epoch >= self.num_epochs:
      raise ValueError("start_itr превышает количество доступных эпох!")

    current_itr = self.start_itr % self.num_samples  

    # Отбрасываем завершенные эпохи
    self.epochs_itrs = self.epochs_itrs[current_epoch:]

    # Отбрасываем уже использованные итерации в текущей эпохе
    self.epochs_itrs[0] = self.epochs_itrs[0][current_itr:]

    # Объединяем все индексы в один тензор
    self.indices = torch.cat(self.epochs_itrs).tolist()

  def __iter__(self):
    """
    Возвращает итератор по индексам выборки.
    
    Возвращает:
    -----------
    Iterator[int]
        Итератор индексов для выборки данных.
    """
    return iter(self.indices)

  def __len__(self):
    """
    Возвращает общее количество оставшихся выборок.
    
    Возвращает:
    -----------
    int
        Длина списка индексов для обучения.
    """
    return len(self.indices)
  

def get_loader(config, 
               batch_size: int, 
               start_itr: int, 
               random_use=False) -> torch.utils.data.DataLoader:
  """
  Создает DataLoader для обучения модели, включая аугментации и кастомный Sampler.
  
  Параметры:
  ----------
  config : dict
      Конфигурационный словарь с параметрами датасета и обучения.
  batch_size : int
      Размер батча данных.
  start_itr : int
      Итерация, с которой начинается обучение (используется для восстановления состояния Sampler).
  random_use : bool, optional
      Если `True`, индексы батчей будут случайными и не сохраняться (по умолчанию `False`).
  
  Возвращает:
  -----------
  torch.utils.data.DataLoader
      DataLoader с кастомным Sampler и предобработанными изображениями.
  """
  transform_list = []

  # Добавляем аугментации, если они включены в конфигурации
  if config['dataset']['use_augments']:
    transform_list.extend([
      transforms.RandomHorizontalFlip(p=0.5),  # Случайное отражение по горизонтали
      transforms.RandomResizedCrop(size=(config['resolution'], config['resolution']), scale=(0.8, 1.0)),  # Рандомное обрезание
      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)  # Цветовые изменения
    ]) 
  else:
    transform_list.append(transforms.Resize((config['resolution'], config['resolution'])))  # Изменение размера без аугментаций
  
  # Преобразование в тензор и нормализация
  transform_list.extend([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])     

  # Создаем датасет с трансформациями
  dataset = Dataset(
    config['dataset']['annot_path'], 
    transforms=transforms.Compose(transform_list)
  ) 

  # Создаем кастомный Sampler для батчей
  sampler = Sampler(
    dataset, 
    num_epochs=config['train']['num_epochs'], 
    start_itr=start_itr, 
    batch_size=batch_size,
    save_path=config['data_root'],
    random_use=random_use
  )

  # Возвращаем DataLoader
  return torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size,
    sampler=sampler, 
    num_workers=min(4, multiprocessing.cpu_count())
  )
