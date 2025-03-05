'''
Файл с реализациями функций потерь (loss functions) для различных вариантов обучения GAN-моделей.
'''

import torch
import torch.nn.functional as F

def loss_dcgan_dis(dis_fake: torch.Tensor, dis_real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Реализация функции потерь дискриминатора в стиле DCGAN.
    Использует функцию softplus для стабильного вычисления логистической потери.
    
    Аргументы:
    ----------
    dis_fake : torch.Tensor
        Оценки дискриминатора на фейковых примерах (сгенерированных).
    dis_real : torch.Tensor
        Оценки дискриминатора на реальных примерах (из датасета).
    
    Возвращает:
    -----------
    tuple[torch.Tensor, torch.Tensor]
        Кортеж из (L1, L2), где:
          - L1 = средняя softplus(-dis_real)
          - L2 = средняя softplus(dis_fake)
    """
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake: torch.Tensor) -> torch.Tensor:
    """
    Реализация функции потерь генератора в стиле DCGAN.
    
    Аргументы:
    ----------
    dis_fake : torch.Tensor
        Оценки дискриминатора на фейковых примерах (сгенерированных).
    
    Возвращает:
    -----------
    torch.Tensor
        Среднее softplus(-dis_fake)
    """
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


def loss_hinge_dis(dis_fake: torch.Tensor, dis_real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Реализация функции потерь дискриминатора в стиле Hinge Loss (самый популярный для BigGAN).
    
    Аргументы:
    ----------
    dis_fake : torch.Tensor
        Оценки дискриминатора на фейковых примерах (сгенерированных).
    dis_real : torch.Tensor
        Оценки дискриминатора на реальных примерах (из датасета).
    
    Возвращает:
    -----------
    tuple[torch.Tensor, torch.Tensor]
        Кортеж (loss_real, loss_fake) — две компоненты для реальных и фейковых.
    """
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake: torch.Tensor) -> torch.Tensor:
    """
    Реализация функции потерь генератора в стиле Hinge Loss.
    G стремится максимизировать оценку дискриминатора на фейках.
    
    Аргументы:
    ----------
    dis_fake : torch.Tensor
        Оценки дискриминатора на фейковых примерах (сгенерированных).
    
    Возвращает:
    -----------
    torch.Tensor
        Hinge Loss для генератора
    """
    loss = -torch.mean(dis_fake)
    return loss


# По умолчанию задаём генератору/дискриминатору hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
