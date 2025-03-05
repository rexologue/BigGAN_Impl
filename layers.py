''' 
layers.py

Этот файл содержит реализации слоев, из которых состоит BigGAN.
'''
  
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_norm import SpectralNorm


##################################
# Spectal Normalized Convolution #
##################################
class SNConv2d(nn.Conv2d, SpectralNorm):
    """
    2D-свёрточный слой (Conv2d) с применением спектральной нормализации (SpectralNorm).

    Спектральная нормализация ограничивает наибольшее сингулярное значение матрицы весов,
    что способствует стабильному обучению и снижает риск "взрыва" градиентов, особенно 
    в генеративных моделях (GAN).

    Механизм:
    ---------
    1. Оценка наибольшего сингулярного значения sv методом Power Iteration.
    2. Нормализация весов путём деления на найденное sv (Weights / sv).

    Параметры:
    -----------
    in_channels : int
        Количество входных каналов.
    out_channels : int
        Количество выходных каналов.
    kernel_size : int or tuple
        Размер ядра свёртки.
    stride : int or tuple, optional
        Шаг свёртки. По умолчанию 1.
    padding : int or tuple, optional
        Паддинг свёртки. По умолчанию 0.
    dilation : int or tuple, optional
        Диляция (dilation). По умолчанию 1.
    groups : int, optional
        Количество групп свёртки. По умолчанию 1.
    bias : bool, optional
        Если True, используется смещение (bias). По умолчанию True.
    num_svs : int, optional
        Количество оцениваемых наибольших сингулярных значений (как правило, 1). По умолчанию 1.
    num_itrs : int, optional
        Количество итераций метода степенных итераций (power iteration). По умолчанию 1.
    eps : float, optional
        Малое число для предотвращения деления на ноль. По умолчанию 1e-12.

    Пример:
    --------
    >>> snconv = SNConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> out = snconv(x)
    >>> out.shape
    torch.Size([4, 64, 32, 32])
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 num_svs=1,
                 num_itrs=1,
                 eps=1e-12):
        
        nn.Conv2d.__init__(self,
                           in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias)
        
        # Инициализируем "миксин" спектральной нормализации
        SpectralNorm.__init__(self,
                              num_svs=num_svs,
                              num_itrs=num_itrs,
                              num_outputs=out_channels,
                              eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Вызов self.W_normalized для получения нормализованных весов (из SpectralNorm)
        return F.conv2d(x,
                        self.W_normalized,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)


#############################
# Spectal Normalized Linear #
#############################
class SNLinear(nn.Linear, SpectralNorm):
    """
    Полносвязный слой (Linear) с применением спектральной нормализации (SpectralNorm).

    Механизм:
    ---------
    1. Оценка наибольшего сингулярного значения sv методом Power Iteration.
    2. Нормализация весов путём деления на найденное sv (Weights / sv).

    Параметры:
    -----------
    in_features : int
        Количество входных нейронов (размер входного вектора).
    out_features : int
        Количество выходных нейронов (размер выходного вектора).
    bias : bool, optional
        Если True, к выходу добавляется смещение (bias). По умолчанию True.
    num_svs : int, optional
        Количество оцениваемых наибольших сингулярных значений. По умолчанию 1.
    num_itrs : int, optional
        Количество итераций метода степенных итераций (power iteration). По умолчанию 1.
    eps : float, optional
        Малое число для предотвращения деления на ноль. По умолчанию 1e-12.

    Пример:
    --------
    >>> snlinear = SNLinear(128, 64)
    >>> x = torch.randn(32, 128)
    >>> out = snlinear(x)
    >>> out.shape
    torch.Size([32, 64])
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 num_svs: int = 1,
                 num_itrs: int = 1,
                 eps: float = 1e-12):
        
        nn.Linear.__init__(self, in_features, out_features, bias)

        SpectralNorm.__init__(self,
                              num_svs=num_svs,
                              num_itrs=num_itrs,
                              num_outputs=out_features,
                              eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Вызов self.W_normalized для получения нормализованных весов (из SpectralNorm)
        return F.linear(x, self.W_normalized, self.bias)


#################################
# Spectal Normalized Embeddings #
#################################
class SNEmbedding(nn.Embedding, SpectralNorm):
  def __init__(self, 
               num_embeddings: int, 
               embedding_dim: int, 
               padding_idx=None, 
               max_norm=None, 
               norm_type=2, 
               scale_grad_by_freq=False,
               sparse=False, 
               _weight=None,
               num_svs=1, 
               num_itrs=1, 
               eps=1e-12):
    """
    SNEmbedding (Спектрально-нормализованный Embedding).

    Механизм:
    ---------
    1. Оценка наибольшего сингулярного значения sv методом Power Iteration.
    2. Нормализация весов путём деления на найденное sv (Weights / sv).

    Параметры:
    -----------
    num_embeddings : int
        Размер словаря (общее число различных векторов).
    embedding_dim : int
        Размерность векторного представления (embedding).
    padding_idx : int, optional
        Индекс токена, используемого для "padding".
    max_norm, norm_type, scale_grad_by_freq, sparse
        Параметры, наследуемые от nn.Embedding, смотрите документацию PyTorch.
    _weight : torch.Tensor, optional
        Явно заданная матрица эмбеддингов.
    num_svs : int
        Количество сингулярных векторов для аппроксимации спектральной нормы.
    num_itrs : int
        Число итераций степенного метода.
    eps : float
        Малое число для численной устойчивости при вычислении SN.

    Примечания:
    -----------
    - Данный класс ожидает, что где-то внутри будет вызван механизм
      обновления сингулярных векторов (обычно в методе forward).
    - В исходном коде BigGAN механизм спектральной нормализации
      реализован в виде отдельного класса SpectralNorm, с которым
      SNEmbedding наследуется совместно.
    """
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    
    SpectralNorm.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

  def forward(self, x):
    return F.embedding(x, self.W_normalized)


#############
# Attention #
#############
class Attention(nn.Module):
    """
    Реализация механизма самовнимания (Self-Attention). 
    Данный модуль позволяет модели "фокусироваться" на разных 
    участках карты признаков, чтобы улавливать дальние зависимости и улучшать 
    качество генерируемых/обрабатываемых изображений.

    Механизм:
    ---------
    1. Преобразование входных активаций в "ключи" (theta), "запросы" (phi) 
       и "значения" (g) с помощью 1×1 свёрток.
    2. Сжатие пространственного разрешения (max_pool2d) для phi и g, чтобы 
       уменьшить вычислительные затраты при расчёте карт внимания (attention map).
    3. Матричное умножение (theta^T * phi) c последующей нормализацией 
       (softmax) даёт карту внимания beta.
    4. Применение карты внимания beta к g восстанавливает пространственное 
       разрешение через обратные reshape и выходную 1×1 свёртку (o).
    5. Результат смешивается со входом x по формуле: x + gamma * o.

    Параметры:
    -----------
    ch : int
        Количество входных каналов карты признаков (feature map).
    which_conv : nn.Module, optional
        Класс свёрточного слоя, используемый для создания сверточных модулей theta, phi, g и o.
        По умолчанию SNConv2d (с использованием спектральной нормализации).
    name : str, optional
        Неиспользуемый в данном коде параметр, сохранён для совместимости.
        По умолчанию 'attention'.

    Атрибуты:
    -----------
    ch : int
        Количество каналов во входном тензоре.
    which_conv : nn.Module
        Класс свёрточного слоя, используемый внутри блока внимания.
    theta : nn.Module
        1×1 свёртка, сжимающая число каналов в 8 раз (ключи).
    phi : nn.Module
        1×1 свёртка, сжимающая число каналов в 8 раз (запросы) + max pool.
    g : nn.Module
        1×1 свёртка, сжимающая число каналов в 2 раза (значения) + max pool.
    o : nn.Module
        1×1 свёртка, восстанавливающая число каналов до ch.
    gamma : torch.nn.Parameter
        Обучаемый коэффициент (скаляр), управляющий вкладом блока внимания.

    Пример:
    --------
    >>> # Предположим, мы хотим использовать спектральную нормализацию
    >>> # и имеем класс SNConv2d, как в вашем коде:
    ...
    >>> attention_block = Attention(ch=64, which_conv=SNConv2d)
    >>> x = torch.randn(4, 64, 32, 32)  # batch_size=4, 64 каналов, 32x32
    >>> out = attention_block(x)
    >>> out.shape
    torch.Size([4, 64, 32, 32])
    """
    def __init__(self, 
                 ch: int, 
                 which_conv=nn.Conv2d):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv

        # Создаём сверточные слои для ключей, запросов, значений и выходов
        self.theta = self.which_conv(in_channels=self.ch,
                                     out_channels=self.ch // 8,
                                     kernel_size=1,
                                     padding=0,
                                     bias=False)

        self.phi = self.which_conv(in_channels=self.ch,
                                   out_channels=self.ch // 8,
                                   kernel_size=1,
                                   padding=0,
                                   bias=False)

        self.g = self.which_conv(in_channels=self.ch,
                                 out_channels=self.ch // 2,
                                 kernel_size=1,
                                 padding=0,
                                 bias=False)

        self.o = self.which_conv(in_channels=self.ch // 2,
                                 out_channels=self.ch,
                                 kernel_size=1,
                                 padding=0,
                                 bias=False)

        # Обучаемый скаляр gamma, регулирующий вклад ветви внимания
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        # 1) Вычисляем ключи (theta), запросы (phi) и значения (g)
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)

        # 2) Изменяем форму тензоров для матричного умножения
        # theta : (batch, ch//8, H*W)
        b, _, h, w = theta.shape
        theta = theta.view(b, self.ch // 8, h * w)

        # phi : (batch, ch//8, (H*W)//4) после pooling
        b, _, h2, w2 = phi.shape
        phi = phi.view(b, self.ch // 8, h2 * w2)

        # g : (batch, ch//2, (H*W)//4) после pooling
        g = g.view(b, self.ch // 2, h2 * w2)

        # 3) Вычисляем карту внимания: beta = softmax( theta^T * phi )
        #    theta^T: (batch, H*W, ch//8)
        #    phi:     (batch, ch//8, (H*W)//4)
        #    torch.bmm - по-батчевое матричное умножение
        beta = torch.bmm(theta.transpose(1, 2), phi)  # (batch, H*W, (H*W)//4)
        beta = F.softmax(beta, dim=-1)

        # 4) Применяем карту внимания к g: (batch, ch//2, (H*W)//4) * (batch, (H*W)//4, H*W)
        o = torch.bmm(g, beta.transpose(1, 2))  # (batch, ch//2, H*W)

        # Возвращаем пространственную форму
        o = o.view(b, self.ch // 2, h, w)
        o = self.o(o)  # (batch, ch, H, W)

        # 5) Смешиваем результат с исходным входом, умножая на gamma
        return x + self.gamma * o
    

#######################
# Batch Normalization #
#######################
class BatchNorm(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1):
    """
    Упрощённая реализация BatchNorm (без условной части).

    Параметры:
    -----------
    output_size : int
        Количество каналов/фильтров.
    eps : float
        Параметр эпсилон (избежание деления на 0).
    momentum : float
        Коэффициент экспоненциального усреднения для running статистик.

    Атрибуты:
    -----------
    gain : nn.Parameter
        Аналог gamma (масштабирующий параметр).
    bias : nn.Parameter
        Аналог beta (смещающий параметр).
    stored_mean : torch.Tensor
        Скользящее среднее активаций.
    stored_var : torch.Tensor
        Скользящая дисперсия активаций.
    """
    super(BatchNorm, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = nn.Parameter(torch.ones(output_size), requires_grad=True)
    self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    
    self.register_buffer('stored_mean', torch.zeros(output_size))
    self.register_buffer('stored_var',  torch.ones(output_size))
        
  def forward(self, x, y=None):
    return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)


################################################################
# Class Conditional Batch Normalization with Shared Embeddings #
################################################################
class ConditionalSharedBatchNorm(nn.Module):
    """
    Реализация условной (conditional) BatchNorm, используемой в BigGAN.
    Данный модуль позволяет "подмешивать" информацию о классе и/или
    дополнительном шуме в нормализацию активаций, тем самым повышая
    выразительность генератора.

    Механизм:
    ---------
    Условная BatchNorm (`BN_cond`) задаётся формулой:
    
    `BN_cond(x, y) = gamma(y) * [ (x - batch_mean) / sqrt(batch_variance + epsilon) ] + beta(y)`,
    
    где:
    - `x` — входные активации (тензор размерности `(batch_size, ch, H, W)`).
    - `batch_mean` и `batch_variance` — среднее и стандартное отклонение (или обученные статистики) для данного батча/слоя.
    - `gamma(y)` и `beta(y)` — условные масштабирующие и смещающие коэффициенты, вычисленные из вектора `y` (конкатенация эмбеддинга класса и кусочка шума).
    - `epsilon` — малое число для избежания деления на ноль.

    С помощью линейных слоёв (`gain` и `bias`) мы генерируем `gamma(y)` и `beta(y)`
    непосредственно на основе входа `y`. Это даёт гибкость генератору
    формировать различные стили и текстуры в зависимости от класса и шума.

    Параметры:
    -----------
    output_size : int
        Количество каналов (`C`), на которых выполняется нормализация.
        Обычно совпадает с числом каналов в входном тензоре `x`.
    input_size : int
        Размер входа для линейных слоёв `gain` и `bias`.
        Например, это может быть сумма размерности эмбеддинга класса
        и выделенной части шума `z_l`.
    which_linear : nn.Module
        Класс линейного слоя, который используется для
        создания модулей `gain` и `bias`. Например, может быть `nn.Linear`.
    eps : float, optional
        Малое число (`epsilon`) для предотвращения деления на ноль в batch-norm.
        По умолчанию 1e-5.
    momentum : float, optional
        Коэффициент для экспоненциального усреднения статистик
        (`stored_mean` и `stored_var`). По умолчанию 0.1.

    Атрибуты:
    -----------
    output_size : int
        Количество каналов для нормализации.
    input_size : int
        Размер входа для линейных слоёв.
    gain : nn.Module
        Линейный слой для вычисления `gamma(y) - 1`.
        В коде используется (1 + self.gain(y)), чтобы начальное значение
        `gamma` было близко к 1.
    bias : nn.Module
        Линейный слой для вычисления `beta(y)`.
    stored_mean : torch.Tensor
        Тензор, хранящий скользящее среднее (running mean) по батчам
        во время обучения (для режима eval/инференса).
    stored_var : torch.Tensor
        Тензор, хранящий скользящую дисперсию (running var).
    eps : float
        Параметр эпсилон (см. выше).
    momentum : float
        Параметр момента (см. выше).

    Пример:
    --------
    >>> # Предположим, что у нас вектор y размерностью 128,
    >>> # а число каналов активаций равно 256
    >>> cond_bn = ConditionalSharedBatchNorm(output_size=256, 
    ...                                      input_size=128, 
    ...                                      which_linear=nn.Linear)

    >>> x = torch.randn(4, 256, 32, 32)   # batch=4, 256 каналов, 32x32
    >>> y = torch.randn(4, 128)           # условный вектор (класс + шум)
    >>> out = cond_bn(x, y)
    >>> out.shape
    torch.Size([4, 256, 32, 32])
    """
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 which_linear: nn.Module,
                 eps: float = 1e-5,
                 momentum: float = 0.1):
        super(ConditionalSharedBatchNorm, self).__init__()

        self.output_size = output_size
        self.input_size = input_size

        # Линейные слои для gamma(y) и beta(y)
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        self.eps = eps
        self.momentum = momentum

        # Регистрируем буферы для хранения статистик по батчу
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 1) Вычисляем gamma(y) и beta(y)
        #    (1 + self.gain(y)) - при инициализации gamma ~ 1.
        #    Форму (batch_size, C, 1, 1) получаем через .view(...)
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        # 2) Применяем стандартную batch_norm (без learnable weight/bias)
        #    Используем stored_mean, stored_var как "running" статистики.
        #    Аргументы: training = self.training
        #               momentum = self.momentum
        #               eps = self.eps
        out = F.batch_norm(
            x,
            self.stored_mean,
            self.stored_var,
            weight=None,
            bias=None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )

        # 3) "Накладываем" класс- (и шум-) зависимые gamma/beta
        out = out * gain + bias
        return out

    def extra_repr(self) -> str:
        """
        Возвращает дополнительную строку, дополняющую отображение модуля
        при печати (print(module)).
        """
        s = f"out: {self.output_size}, in: {self.input_size}"
        return s


###################
# Generator block #
###################
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv, which_bn, activation_fn=None, 
                 upsample=None):
        """
        Residual block для генератора в BigGAN, 
        поддерживающий условную нормализацию (Conditional BatchNorm) и 
        возможное увеличение разрешения (upsample).

        Механизм:
        ---------
        1. К входным активациям `x` последовательно применяются:
          - Условная BatchNorm `bn1` с учётом вектора `y` (например, класс + шум).
          - Функция активации (по умолчанию ReLU).
          - (Опционально) увеличение разрешения (upsample), если задано.
          - Свёртка `conv1`, приводящая число каналов от `in_channels` к `out_channels`.
          - Снова условная BatchNorm `bn2` + активация.
          - Ещё одна свёртка `conv2`, сохраняющая размер каналов на уровне `out_channels`.
        2. Параллельно shortcut-ветка может включать:
          - (Опционально) upsample для согласования разрешения с основной веткой.
          - (Опционально) 1×1 свёртку, если нужно подогнать число каналов 
            (или используется нестандартный подход, когда любой upsample влечёт learnable shortcut).
        3. На выходе сумма основной и shortcut-ветвей формирует выход блока.

        Параметры:
        -----------
        in_channels : int
            Количество каналов во входном тензоре.
        out_channels : int
            Количество каналов в выходном тензоре (после свёрток).
        which_conv : nn.Module
            Класс свёрточного слоя (например, `nn.Conv2d` или `SNConv2d`), 
            используемый в блоке.
        which_bn : nn.Module
            Класс условной BatchNorm (например, `ConditionalSharedBatchNorm`), 
            который будет применяться для нормализации с учётом `y`.
        activation_fn : callable, optional
            Функция активации (например, `F.relu`). Если не задана, 
            может использоваться ReLU по умолчанию.
        upsample : callable или None, optional
            Функция/операция для увеличения разрешения (например, `nn.Upsample`). 
            Если `None`, апсемплинг не выполняется.

        Атрибуты:
        -----------
        in_channels : int
            Число каналов на входе блока.
        out_channels : int
            Число каналов на выходе блока.
        conv1 : nn.Module
            Первая свёртка, переводящая `in_channels` → `out_channels`.
        conv2 : nn.Module
            Вторая свёртка, работающая уже на `out_channels`.
        shortcut : nn.Module
            Операция shortcut (skip-ветка). Может быть `1×1` свёрткой или `nn.Identity`.
        bn1 : nn.Module
            Условная BatchNorm, применяемая до первой свёртки.
        bn2 : nn.Module
            Условная BatchNorm, применяемая до второй свёртки.
        activation_fn : callable
            Функция активации, вызываемая после BatchNorm.
        upsample : callable or None
            Ссылка на функцию/операцию апсемплинга.

        Пример:
        --------
        >>> # Предположим, что у нас:
        >>> # in_channels=128, out_channels=256,
        >>> # условная BN из ConditionalSharedBatchNorm
        >>> # и upsample - это какая-то функция из nn.Upsample
        >>> block = GBlock(
        ...     in_channels=128,
        ...     out_channels=256,
        ...     which_conv=nn.Conv2d,
        ...     which_bn=ConditionalSharedBatchNorm,
        ...     activation_fn=F.relu,
        ...     upsample=nn.Upsample(scale_factor=2)
        ... )
        >>> x = torch.randn(8, 128, 16, 16)  # batch=8, 128 каналов, 16x16
        >>> y = torch.randn(8, 100)         # условный вектор
        >>> out = block(x, y)
        >>> out.shape
        torch.Size([8, 256, 32, 32])
        """
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation_fn = activation_fn     # Функция активации (обычно ReLU)
        self.upsample = upsample         # Функция или слой для увеличения разрешения (может быть None)

        # 1) Свёрточные слои
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)

        # 2) Логика shortcut-соединения
        #    Если число входных и выходных каналов не совпадает
        #    или если нужно увеличить разрешение, то Shortcut (skip) тоже нужно «привести» по размеру:
        if (in_channels != out_channels) or (upsample is not None):
            self.shortcut = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        else:
           self.shortcut = nn.Identity()

        # 3) BatchNorm слои (условные)
        #    which_bn обычно указывает на ConditionalSharedBatchNorm или похожий класс,
        #    чтобы нормализация могла учитывать y (условный вектор/эмбеддинг).
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)

        # 4) Сохранение ссылки на upsample-операцию
        self.upsample = upsample

    def forward(self, x, y):
        # ----------------- Основная ветка (h) --------------------
        # (a) Сначала BatchNorm + активация
        h = self.activation_fn(self.bn1(x, y))

        # (b) Если есть upsample-функция, то увеличиваем разрешение
        if self.upsample is not None:
            h = self.upsample(h)
            # Чтобы сложить с skip-веткой, нужно поднять разрешение входа
            x = self.upsample(x)

        # (c) Применяем первую свёртку
        h = self.conv1(h)

        # (d) Снова BatchNorm + активация
        h = self.activation_fn(self.bn2(h, y))

        # (e) Вторая свёртка
        h = self.conv2(h)

        # ----------------- Shortcut (skip) ветка -----------------
        x = self.shortcut(x)

        # Сложение главной и skip-ветвей (residual connection)
        return h + x


#######################
# Discriminator block #
#######################
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=SNConv2d, wide=True,
                 preactivation=False, activation_fn=None, downsample=None,):
        """
        Residual block для дискриминатора в BigGAN, 
        поддерживающий уменьшение разрешения (downsample) и 
        спектральную нормализацию (SN).

        Механизм:
        ---------
        1. (Опционально) Если `preactivation=True`, применяется ReLU к входу до первой свёртки.
        2. Первая свёртка `conv1` с числом каналов от `in_channels` к `hidden_channels`.
          - При `wide=True`, `hidden_channels == out_channels`.
          - При `wide=False`, `hidden_channels == in_channels`.
        3. Применяем активацию (например, ReLU), затем вторую свёртку `conv2`.
        4. (Опционально) Уменьшение разрешения (downsample).
        5. Shortcut-ветка выполняет те же операции (conv и/или downsample), если 
          в блоке есть изменение числа каналов или downsample.
        6. Результаты основной и shortcut-ветвей складываются.

        Параметры:
        -----------
        in_channels : int
            Количество каналов во входном тензоре.
        out_channels : int
            Количество каналов в выходном тензоре (после свёрток).
        which_conv : nn.Module, optional
            Класс свёрточного слоя, часто `SNConv2d`, включающий спектральную нормализацию.
        wide : bool, optional
            Если True, промежуточный блок (между conv1 и conv2) имеет ширину `out_channels`.
            Если False, промежуточный блок имеет ширину `in_channels`.
        preactivation : bool, optional
            Если True, это «pre-act ResNet»-подход, когда ReLU вызывается до conv1. 
            Если False, активация идёт между conv1 и conv2 (после conv1).
        activation_fn : callable, optional
            Функция активации (например, `F.relu`).
        downsample : callable or None, optional
            Операция для уменьшения разрешения (например, `nn.AvgPool2d` или `nn.Conv2d` со stride>1). 
            Если None, не происходит downsample.

        Атрибуты:
        -----------
        in_channels : int
            Число каналов на входе блока.
        out_channels : int
            Число каналов на выходе блока.
        hidden_channels : int
            Число каналов после первой свёртки (определяется `wide`).
        conv1 : nn.Module
            Первая свёрточная операция (in_channels → hidden_channels).
        conv2 : nn.Module
            Вторая свёрточная операция (hidden_channels → out_channels).
        shortcut_conv : nn.Module или nn.Identity
            Свёртка (1×1) для skip-ветки, если нужно изменить число каналов.
        downsample : callable or None
            Ссылка на операцию уменьшения разрешения.
        preactivation : bool
            Логический флаг, влияющий на порядок активации.
        activation_fn : callable
            Функция активации (например, `F.relu`).

        Пример:
        --------
        >>> # Предположим, что у нас:
        >>> # in_channels=256, out_channels=128,
        >>> # wide=True, preactivation=False
        >>> block = DBlock(
        ...     in_channels=256,
        ...     out_channels=128,
        ...     which_conv=SNConv2d, 
        ...     wide=True,
        ...     preactivation=False,
        ...     activation_fn=F.relu,
        ...     downsample=nn.AvgPool2d(kernel_size=2)
        ... )
        >>> x = torch.randn(8, 256, 64, 64)  # batch=8, 256 каналов, 64x64
        >>> out = block(x)
        >>> out.shape
        torch.Size([8, 128, 32, 32])
        """
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        # wide=True означает, что скрытый промежуточный блок имеет ширину out_channels (это из BigGAN/SA-GAN).
        # wide=False оставляет скрытый промежуток на уровне in_channels.
        self.hidden_channels = self.out_channels if wide else self.in_channels

        self.which_conv    = which_conv             # Обычно SNConv2d (со спектральной нормализацией)
        self.preactivation = preactivation          # Если True, будет "pre-act ResNet" стиль
        self.activation_fn = activation_fn          # Функция активации
        self.downsample    = downsample             # Функция/операция для уменьшения разрешения (может быть None)

        # 1) Два свёрточных слоя: conv1 и conv2
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)

        # 2) Shortcut (skip) связь:
        #    Нужно приводить каналы и/или понижать разрешение, если:
        #       - in_channels != out_channels
        #       - есть downsample
        if (in_channels != out_channels) or (downsample is not None):
            self.shortcut_conv = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        else:
            self.shortcut_conv = nn.Identity()

    def shortcut(self, x):
        # Метод shortcut выполняет логику преобразования «skip-ветки» 
        # с учётом порядка действий при preactivation=True или False.
        if self.preactivation:
            # preactivation => сначала conv_sc, потом downsample
            x = self.shortcut_conv(x)

            if self.downsample:
                x = self.downsample(x)
        else:
            # postactivation => сначала downsample, потом conv_sc
            if self.downsample:
                x = self.downsample(x)

            x = self.shortcut_conv(x)

        return x

    def forward(self, x):
        # Если preactivation=True, нужно сделать ReLU до первой свёртки
        if self.preactivation:
            # Здесь ReLU применяют «вне места» (out-of-place),
            # чтобы не затирать исходный x для shortcut.
            h = F.relu(x)
        else:
            # Если нет preactivation, то просто передаём x дальше
            h = x

        # 1) Первая свёртка
        h = self.conv1(h)

        # 2) Вызов self.activation (например, ReLU), затем вторая свёртка
        h = self.conv2(self.activation_fn(h))

        # 3) Если downsample не None, уменьшаем разрешение
        if self.downsample:
            h = self.downsample(h)

        # 4) Складываем с shortcut-веткой
        return h + self.shortcut(x)
    