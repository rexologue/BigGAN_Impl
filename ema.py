import torch

class EMA:
  """
  Класс для экспоненциального сглаживания (Exponential Moving Average, EMA) параметров модели.
  
  EMA используется для стабилизации обучения, сохраняя усредненные параметры модели, 
  которые обновляются с заданным коэффициентом затухания (decay).
  
  Основные возможности:
  ----------------------
  - Инициализирует целевую модель параметрами исходной модели.
  - Обновляет параметры целевой модели с заданным коэффициентом затухания.
  - Позволяет начинать EMA после определенной итерации.
  
  Параметры:
  ----------
  source : nn.Module
      Исходная модель, параметры которой будут сглажены.
  target : nn.Module
      Целевая модель, параметры которой будут обновляться усреднением.
  decay : float, optional
      Коэффициент затухания (по умолчанию `0.9999`).
  start_itr : int, optional
      Итерация, с которой начнется обновление EMA (по умолчанию `0`).
  """
  def __init__(self, 
                source: torch.nn.Module, 
                target: torch.nn.Module, 
                decay=0.9999, 
                start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    self.start_itr = start_itr  # Итерация, с которой начинается EMA

  def update(self, itr=None):
    """
    Обновляет параметры целевой модели EMA.
    
    Если передана текущая итерация `itr`, обновление начинается только после `start_itr`.
    
    Параметры:
    ----------
    itr : int, optional
        Текущая итерация обучения. Если `itr < start_itr`, обновление не выполняется.
    """
    if itr is not None and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay

    with torch.no_grad():
      # Берем актуальный state_dict от self.source и self.target на каждой итерации
      source_dict = self.source.state_dict()
      target_dict = self.target.state_dict()
      for key in source_dict:
        target_dict[key].data.copy_(
          target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

    # Обновляем 
    self.target.load_state_dict(target_dict)

    
