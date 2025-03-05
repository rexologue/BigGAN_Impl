''' Spectral Norm
    This file contains implementation of spectral normalization.
'''
import torch
from math_functions import power_iteration

from typing import List


# Spectral normalization base class 
class SpectralNorm:
    """
    Базовый класс для спектральной нормализации (Spectral Normalization).

    Спектральная нормализация — это метод, используемый для стабилизации обучения 
    нейронных сетей, особенно в задачах, связанных с генеративными моделями (например, GAN). 
    Основная идея заключается в ограничении спектральной нормы (максимального сингулярного значения) 
    весов матрицы, что предотвращает неконтролируемый рост градиентов и улучшает устойчивость модели.

    Принцип работы:
    ---------------
    1. Вычисление сингулярных значений:
       - Для весов матрицы W вычисляются сингулярные значения (обычно одно — наибольшее)
         с использованием метода степенных итераций (power iteration).

    2. Нормализация весов:
       - Веса матрицы W делятся на наибольшее сингулярное значение, что ограничивает 
         спектральную норму матрицы значением 1.
       - Это предотвращает "взрыв" градиентов и стабилизирует обучение.

    Применение:
    -----------
    Данный класс не используется самостоятельно, а выступает в качестве "микса" (mixin) 
    к стандартным модулям PyTorch (например, `nn.Conv2d` или `nn.Linear`). 
    Чтобы воспользоваться спектральной нормализацией, нужно унаследовать класс 
    от двух родительских классов одновременно:

    .. code-block:: python

        class SNConv2d(nn.Conv2d, SpectralNorm):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True,
                         num_svs=1, num_itrs=1, eps=1e-12):
                nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias)
                SpectralNorm.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

            def forward(self, x):
                # Вместо self.weight используем self.W_normalized при свертке
                return F.conv2d(x, self.W_normalized, self.bias, self.stride,
                                 self.padding, self.dilation, self.groups)

    Аналогичным образом можно написать SNLinear для линейного слоя.

    Параметры:
    ----------
    num_svs : int
        Количество сингулярных значений, которые будут отслеживаться. 
        Обычно используется только наибольшее сингулярное значение (1).
    num_itrs : int
        Количество итераций метода степенных итераций для оценки сингулярных значений 
        на каждом вызове.
    num_outputs : int
        Количество выходных каналов (или размерность выходного пространства) 
        для матрицы весов (обычно совпадает с out_channels для Conv или out_features для Linear).
    transpose : bool, optional
        Флаг, указывающий, следует ли транспонировать матрицу весов перед вычислением 
        сингулярных значений. По умолчанию False.
    eps : float, optional
        Малое значение, используемое для предотвращения деления на ноль при нормализации. 
        По умолчанию 1e-12.

    Атрибуты:
    ---------
    u : List[torch.Tensor]
        Список левых сингулярных векторов, используемых для оценки сингулярных значений.
    sv : List[torch.Tensor]
        Список сингулярных значений. Эти значения используются только для логирования 
        и не влияют на само обучение.
    W_normalized : torch.Tensor
        Спектрально-нормализованная версия матрицы весов (property). Именно её следует 
        использовать в методе forward вместо self.weight.
    """
    def __init__(self, 
                 num_svs: int, 
                 num_itrs: int, 
                 num_outputs: int, 
                 transpose: bool = False, 
                 eps: float = 1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps

        # Регистрируем сингулярные вектора и значения в буферах.
        for i in range(self.num_svs):
            self.register_buffer(f'u{i}', torch.randn(1, num_outputs))
            self.register_buffer(f'sv{i}', torch.ones(1))

    # Эти методы ожидаются при использовании этого класса вместе с nn.Module.
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Заглушка, чтобы показать идею регистрации буферов.
        На практике вы будете использовать метод nn.Module.register_buffer()
        в собственном классе, унаследованном от nn.Module и SpectralNorm.
        """
        setattr(self, name, tensor)

    @property
    def u(self) -> List[torch.Tensor]:
        """
        Возвращает список левых сингулярных векторов.

        Возвращает:
        -----------
        List[torch.Tensor]
            Список тензоров, каждый из которых представляет левый сингулярный вектор.
        """
        return [getattr(self, f'u{i}') for i in range(self.num_svs)]

    @property
    def sv(self) -> List[torch.Tensor]:
        """
        Возвращает список сингулярных значений.

        Примечание:
        ----------
        Эти значения используются только для логирования и не влияют на обучение.

        Возвращает:
        -----------
        List[torch.Tensor]
            Список тензоров, каждый из которых представляет сингулярное значение.
        """
        return [getattr(self, f'sv{i}') for i in range(self.num_svs)]

    @property
    def W_normalized(self) -> torch.Tensor:
        """
        Вычисляет спектрально-нормализованную версию матрицы весов.

        Процесс:
        -------
        1. Матрица весов преобразуется в двумерный тензор (view).
        2. Применяется метод степенных итераций (power_iteration) для 
           оценки наибольшего сингулярного значения (svs[0]).
        3. Веса нормализуются путём деления на найденное значение svs[0].

        Возвращает:
        -----------
        torch.Tensor
            Спектрально-нормализованная матрица весов.
        """
        # Предполагается, что self.weight определён в классе-потомке, 
        # например, nn.Conv2d или nn.Linear.
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()

        # Выполняем степенной метод несколько раз
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps
            )

        # Обновляем значения сингулярных величин для логирования
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv

        # Возвращаем веса, поделённые на наибольшее сингулярное значение
        return self.weight / svs[0]
