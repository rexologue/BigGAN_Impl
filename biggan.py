import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

import layers


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64'):
  """
  Возвращает словарь с конфигурацией для генератора:
  - in_channels и out_channels: указывают сколько каналов на каждом этапе.
  - upsample: на каких этапах будет происходить увеличение разрешения.
  - resolution: список разрешений (H=8,16,32,...).
  - attention: словарь {разрешение: bool}, где True означает, что
    на данном разрешении подключается блок внимания.

  Параметры:
  -----------
  ch : int
      Базовый множитель числа каналов.
  attention : str
      Строка, указывающая на каком разрешении включать внимание.
      Пример: '64' или '32_64'.

  Возвращает:
  -----------
  dict
      Словарь вида:
      arch[<resolution>] = {
          'in_channels': [...],
          'out_channels': [...],
          'upsample': [...],
          'resolution': [...],
          'attention': {...}
      }
  """
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch


class Generator(nn.Module):
  def __init__(self, 
               G_ch=64, 
               dim_z=128, 
               initial_resolution=4, 
               resolution=128,
               resolution_to_apply_attention='64', 
               n_classes=1000,
               num_G_SVs=1, 
               num_G_SV_itrs=1, 
               embeddings_dim=120,
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12,
               G_init='ortho', 
               skip_init=False, 
               no_optim=False,
               use_sn=True):
    """
    Generator (BigGAN-style).

    Механизм:
    ---------
    1) Принимает латентный вектор z и класс (через эмбеддинг y).
    2) Разбивает z на несколько кусков (num_slots).
       Первый кусок идёт в линейный слой -> тензор начального разрешения.
       Остальные куски конкатенируются с эмбеддингом y для GBlock.
    3) Последовательно проходит через блоки (GBlock), каждый из которых
       может делать upsample и, опционально, подключать слой Self-Attention.
    4) Выдаёт RGB-изображение через финальный conv.

    Параметры:
    -----------
    G_ch : int
        Базовый множитель числа каналов.
    dim_z : int
        Размерность входного шума z.
    initial_resolution : int
        Начальное пространственное разрешение (H = W = initial_resolution),
        на котором будет сформирован тензор после первого линейного слоя.
    resolution : int
        Итоговое разрешение выходного изображения (например, 128x128).
    resolution_to_apply_attention : str
        На каких разрешениях применять Self-Attention (например, '64').
    n_classes : int
        Количество классов для условной генерации.
    embeddings_dim : int
        Размер эмбеддинга, соответствующего каждому классу.
    G_lr, G_B1, G_B2, adam_eps : float
        Параметры оптимизатора Adam.
    BN_eps : float
        Эпсилон для BatchNorm.
    SN_eps : float
        Эпсилон для спектральной нормализации (непосредственно при создании слоёв).
    G_init : str
        Способ инициализации весов ('ortho', 'N02', 'glorot' и т.д.).
    skip_init : bool
        Если True, то не инициализировать явно веса (для отладки).
    no_optim : bool
        Если True, то не создаём self.optim (например, для EMA-копии).
    use_sn : bool
        Признак использования спектральной нормализации.

    Атрибуты:
    -----------
    arch : dict
        Архитектура (список in_channels/out_channels/...).
    num_slots : int
        Количество кусков, на которые делится z (число блоков + 1).
    z_chunk_size : int
        Размер каждого куска z.
    shared : nn.Module
        Эмбеддинг для классов (обычно nn.Embedding).
    blocks : nn.ModuleList
        Последовательность блоков GBlock (и возможно Attention).
    output_layer : nn.Sequential
        Финальный слой: batchnorm -> ReLU -> conv -> Tanh.
    optim : optim.Optimizer
        Оптимизатор (Adam) для параметров генератора.
    """
    
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.initial_resolution = initial_resolution
    # Resolution of the output
    self.resolution = resolution
    # Attention?
    self.resolution_to_apply_attention = resolution_to_apply_attention
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Dimensionality of the embeddings?
    self.embeddings_dim = embeddings_dim
    # nonlinearity for residual blocks
    self.activation_fn = nn.ReLU(inplace=False)
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.use_sn = use_sn
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps

    # Architecture dict
    self.arch = G_arch(self.ch, self.resolution_to_apply_attention)[resolution]

    # Splitting Z on chunks
    # We have len(self.arch['in_channels']) convolutional layers, where we put parts of Z
    # and 1 part for first linear layer
    self.num_slots = len(self.arch['in_channels']) + 1
    self.z_chunk_size = (self.dim_z // self.num_slots)

    # Recalculate latent dimensionality for correct splitting into chunks
    self.dim_z = self.z_chunk_size * self.num_slots

    # Which convs, batchnorms, and linear layers to use
    if self.use_sn:
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
    
    self.which_bn = functools.partial(layers.ConditionalSharedBatchNorm,
                          which_linear=functools.partial(self.which_linear, bias=False),
                          input_size=(self.embeddings_dim + self.z_chunk_size),
                          eps=self.BN_eps)

    #############################################################################
    #                               PREPARE MODEL                               #
    #############################################################################

    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    self.shared = (self.which_embedding(n_classes, self.embeddings_dim))

    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.initial_resolution **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation_fn=self.activation_fn,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(
      layers.BatchNorm(self.arch['out_channels'][-1]),
      self.activation_fn,
      self.which_conv(self.arch['out_channels'][-1], 3)
    )

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if not no_optim:
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps

        self.optim = optim.Adam(
          params=self.parameters(), 
          lr=self.lr,
          betas=(self.B1, self.B2), 
          weight_decay=0,
          eps=self.adam_eps
        )

  # Initialize
  def init_weights(self):
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
          

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # Split z on chunks
    zs = torch.split(z, self.z_chunk_size, 1)
    # Concat, except first, all chunks with class embedding
    ys = [torch.cat([y, item], 1) for item in zs[1:]]
    # Initial Z
    z = zs[0]
      
    # First linear layer
    h = self.linear(z)
    # Reshape to (batch_size, ch, H, W)
    h = h.view(h.size(0), -1, self.initial_resolution, self.initial_resolution)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


def D_arch(ch=64, attention='64'):
  """
  Возвращает словарь с конфигурацией для генератора:
  - in_channels и out_channels: указывают сколько каналов на каждом этапе.
  - downsample: на каких этапах будет происходить уменьшение разрешения.
  - resolution: список разрешений (H=1,2,4,...).
  - attention: словарь {разрешение: bool}, где True означает, что
    на данном разрешении подключается блок внимания.

  Параметры:
  -----------
  ch : int
      Базовый множитель числа каналов.
  attention : str
      Строка, указывающая на каком разрешении включать внимание.
      Пример: '64' или '32_64'.

  Возвращает:
  -----------
  dict
      Словарь вида:
      arch[<resolution>] = {
          'in_channels': [...],
          'out_channels': [...],
          'downsample': [...],
          'resolution': [...],
          'attention': {...}
      }
  """
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch


class Discriminator(nn.Module):

  def __init__(self, 
               D_ch=64, 
               D_wide=True, 
               resolution=128,
               resolution_to_apply_attention='64', 
               n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1,
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8, SN_eps=1e-12, 
               output_dim=1, 
               D_init='ortho', 
               skip_init=False,
               no_optim=False):
    """
    Discriminator (BigGAN-style).

    Механизм:
    ---------
    1) Принимает входное изображение x (размером resolution x resolution)
       и класс y (или метку, используемую в Projection Discriminator).
    2) Последовательно пропускает x через DBlock, уменьшая разрешение и увеличивая каналы.
       При этом может быть задействован Self-Attention (на соответствующих разрешениях).
    3) В конце выполняется глобальный pooling (суммирование h по spatial-осям).
    4) Вычисляет скаляр out = lin(h) + sum(embed(y)*h), где embed(y) — проекция
       класса на финальные признаки, реализуя так называемый Projection Discriminator.

    Параметры:
    -----------
    D_ch : int
        Базовый множитель числа каналов.
    D_wide : bool
        Флаг «широкого» дискриминатора.
    resolution : int
        Разрешение входа (напр. 128).
    resolution_to_apply_attention : str
        На каких разрешениях включать Self-Attention.
    n_classes : int
        Количество классов (для проекции).
    num_D_SVs, num_D_SV_itrs : int
        Параметры спектральной нормализации.
    D_lr, D_B1, D_B2, adam_eps : float
        Параметры оптимизатора.
    SN_eps : float
        Эпсилон для SN.
    output_dim : int
        Размерность финального выхода (по умолчанию 1).
    D_init : str
        Способ инициализации весов.
    skip_init : bool
        Если True, пропускать инициализацию весов.
    """
    super(Discriminator, self).__init__()

    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Attention?
    self.resolution_to_apply_attention = resolution_to_apply_attention
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation_fn = nn.ReLU(inplace=False)
    # Initialization style
    self.init = D_init
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Architecture
    self.arch = D_arch(self.ch, self.resolution_to_apply_attention)[resolution]

    self.which_conv = functools.partial(layers.SNConv2d,
                        kernel_size=3, padding=1,
                        num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                        eps=self.SN_eps)
    self.which_linear = functools.partial(layers.SNLinear,
                        num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                        eps=self.SN_eps)
    self.which_embedding = functools.partial(layers.SNEmbedding,
                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                            eps=self.SN_eps)
    
    #############################################################################
    #                               PREPARE MODEL                               #
    #############################################################################

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation_fn=self.activation_fn,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    if not no_optim:
      self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps

      self.optim = optim.Adam(
        params=self.parameters(), 
        lr=self.lr,
        betas=(self.B1, self.B2), 
        weight_decay=0, 
        eps=self.adam_eps
      )
    

  # Initialize
  def init_weights(self):
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')


  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x

    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)

    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation_fn(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

    return out


class GAN(nn.Module):
  def __init__(self, G: Generator, D: Discriminator):
    """
    Generative Adversarial Network.
    Класс для упрощения взаимодействия с генератором и дискриминатором при обучении.

    Механизм:
    ---------
    1) Определяем, шаг обучения генератора или дискриминатора происходит в данный момент времени.
    2) Последовательно пропускает x через DBlock, уменьшая разрешение и увеличивая каналы.
       При этом может быть задействован Self-Attention (на соответствующих разрешениях).
    3) В конце выполняется глобальный pooling (суммирование h по spatial-осям).
    4) Вычисляет скаляр out = lin(h) + sum(embed(y)*h), где embed(y) — проекция
       класса на финальные признаки, реализуя так называемый Projection Discriminator.

    Параметры:
    -----------
    z: torch.Tensor
      Вектор латентного пространства для генератора.
    gy: torch.Tensor
      Метки для генератора.
    x: torch.Tensor
      Реальные изображения.
    dy: torch.Tensor
      Метки реальных изображений.
    train_G: bool
      Флаг, обозначающий, учится ли на этом шаге генератор.
    return_G_z: bool
      Флаг, обозначающий, нужно ли возращать результат генерации.
    """
    super(GAN, self).__init__()
    self.G = G
    self.D = D


  def forward_G(self, z, gy):
    with torch.set_grad_enabled(True):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))

    D_fake = self.D(G_z, gy)

    return D_fake
    

  def forward_D(self, z, gy, x, dy):    
    # If training G, enable grad tape
    with torch.set_grad_enabled(False):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))

    D_input = torch.cat([G_z, x], 0)
    D_class = torch.cat([gy, dy], 0)

    # Get Discriminator output
    D_out = self.D(D_input, D_class)

    D_fake, D_real = torch.split(D_out, [G_z.shape[0], x.shape[0]])
    
    return D_fake, D_real 
  