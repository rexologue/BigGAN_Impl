import torch
import torch.nn.functional as F

from typing import List, Tuple


def proj(x: torch.Tensor, 
         y: torch.Tensor) -> torch.Tensor:
    """
    Проекция вектора x на вектор y.

    Проекция вектора x на вектор y вычисляется по формуле:
    proj_y(x) = (y^T * x) / (y^T * y) * y

    Параметры:
    ----------
    x : torch.Tensor
        Входной вектор, который проецируется на вектор y. Размерность: (1, n)
    y : torch.Tensor
        Вектор, на который проецируется x. Размерность: (1, n)

    Возвращает:
    -----------
    torch.Tensor
        Вектор, являющийся проекцией x на y. Размерность: (1, n)

    Пример:
    --------
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> y = torch.tensor([[1.0, 0.0, 0.0]])
    >>> proj(x, y)
    tensor([[1., 0., 0.]])
    """
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x: torch.Tensor, 
                 ys: List[torch.Tensor]) -> torch.Tensor:
    """
    Ортогонализация вектора x относительно списка векторов ys 
    с использованием процесса Грама-Шмидта.

    Процесс Грама-Шмидта последовательно вычитает проекции вектора x 
    на каждый из векторов в списке ys, чтобы получить вектор, 
    ортогональный всем векторам в ys.

    Параметры:
    ----------
    x : torch.Tensor
        Входной вектор, который необходимо ортогонализовать. Размерность: (1, n)
    ys : List[torch.Tensor]
        Список векторов, относительно которых ортогонализуется x. 
        Каждый вектор в списке имеет размерность: (1, n)

    Возвращает:
    -----------
    torch.Tensor
        Вектор, ортогонализованный относительно всех векторов в ys. 
        Размерность: (1, n)

    Пример:
    --------
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> ys = [
    ...     torch.tensor([[1.0, 0.0, 0.0]]), 
    ...     torch.tensor([[0.0, 1.0, 0.0]])
    ... ]
    >>> gram_schmidt(x, ys)
    tensor([[0., 0., 3.]])
    """
    for y in ys:
        x = x - proj(x, y)

    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W: torch.Tensor, 
                    u_: List[torch.Tensor], 
                    update: bool = True, 
                    eps: float = 1e-12
                   ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Применение метода степенных итераций для оценки топ-N сингулярных значений 
    и соответствующих сингулярных векторов матрицы W.

    Метод степенных итераций используется для нахождения наибольших сингулярных значений
    и соответствующих сингулярных векторов матрицы. В данном случае метод применяется
    для оценки топ-N сингулярных значений и векторов, если в списке u_ несколько векторов.

    Параметры:
    ----------
    W : torch.Tensor
        Матрица, для которой оцениваются сингулярные значения и векторы. Размерность: (m, n)
    u_ : List[torch.Tensor]
        Список начальных приближений для левых сингулярных векторов. 
        Каждый вектор имеет размерность: (1, m)
    update : bool, optional
        Флаг, указывающий, следует ли обновлять начальные приближения u_ на каждой итерации. 
        По умолчанию True.
    eps : float, optional
        Малое значение, используемое для нормализации векторов и предотвращения 
        деления на ноль. По умолчанию 1e-12.

    Возвращает:
    -----------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        Кортеж, содержащий три списка:
        1. Список сингулярных значений (torch.Tensor).
        2. Список левых сингулярных векторов (torch.Tensor).
        3. Список правых сингулярных векторов (torch.Tensor).

    Пример:
    --------
    >>> W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> u_ = [torch.tensor([[1.0, 0.0]])]  # (1 x 2)
    >>> svs, us, vs = power_iteration(W, u_)
    >>> svs
    [tensor(5.4649)]
    >>> us  # левые сингулярные вектора
    [tensor([[0.5760, 0.8174]])]
    >>> vs  # правые сингулярные вектора
    [tensor([[0.4046, 0.9145]])]
    """
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Одно обновление Power Iteration
        with torch.no_grad():
            # Получаем правый сингулярный вектор v
            v = torch.matmul(u, W)
            # Ортогонализуем v к уже найденным vs
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]

            # Обновляем левый сингулярный вектор u
            u = torch.matmul(v, W.t())
            # Ортогонализуем u к уже найденным us
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]

            # При необходимости сохраняем новое значение вектора u
            if update:
                u_[i][:] = u

    # Вычисляем сингулярное значение (первое из списка)
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]

    return svs, us, vs


def orthogonal_regularization(model: torch.nn.Module, strength=1e-4, blacklist=[]):
    """
    Применяет модифицированную ортогональную регуляризацию к параметрам модели, напрямую вычисляя градиенты.

    Регуляризация стимулирует ортогональность строк матричных параметров, добавляя соответствующий градиент
    к параметрам модели. Метод работает напрямую с градиентами, избегая явного вычисления потерь.

    Параметры:
    ----------
    model : torch.nn.Module
        Нейронная сеть, к параметрам которой применяется регуляризация.
    strength : float, optional
        Коэффициент силы регуляризации. Определяет вес градиента регуляризации. 
        По умолчанию 1e-4.
    blacklist : List[torch.Tensor], optional
        Список параметров модели, которые следует исключить из регуляризации.
        По умолчанию пустой список.

    Возвращает:
    -----------
    None
        Функция модифицирует градиенты параметров модели in-place.

    Примечания:
    -----------
    - Применяется только к параметрам с размерностью ≥2 (матрицы и выше).
    - Градиент вычисляется по формуле:
        grad = 2 * ((W @ W.T) * (1 - I)) @ W,
      где W - матрица параметра (reshape в 2D), I - единичная матрица.
    - Регуляризация штрафует скалярные произведения между разными строками матриц параметров,
      стимулируя их ортогональность.

    Пример использования:
    ----------------------
    >>> model = torch.nn.Linear(20, 30)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    >>> 
    >>> # Внутри цикла обучения:
    >>> outputs = model(inputs)
    >>> loss = criterion(outputs, targets)
    >>> loss.backward()          # Вычисляем основные градиенты
    >>> ortho(model, strength=1e-4)  # Добавляем градиенты регуляризации
    >>> optimizer.step()        # Обновляем параметры
    >>> optimizer.zero_grad()
    """
    with torch.no_grad():
        for param in model.parameters():
            # Пропускаем параметры с < 2 осями или в blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            
            # Преобразовываем к размерности [batch_size, all_other_dimensions]
            w = param.view(param.shape[0], -1)

            # Считаем градиент
            grad = (2 * torch.mm(torch.mm(w, w.t()) 
                    * (1. - torch.eye(w.shape[0], device=w.device)), w))
            
            # Сразу обновляем веса
            param.grad.data += strength * grad.view(param.shape)


def covariance_matrix(m: torch.Tensor, 
                      rowvar=False) -> torch.Tensor:
    """
    Оценка матрицы ковариации для заданных данных.

    Ковариация указывает уровень, с которым две переменные изменяются совместно.
    Если рассмотреть `N`-мерные наблюдения, `X = [x_1, x_2, ... x_N]^T`,
    то элемент матрицы ковариации `C_{ij}` представляет собой ковариацию
    между переменными `x_i` и `x_j`. Элемент `C_{ii}` является дисперсией переменной `x_i`.

    Аргументы:
    ---------
        `m`: одномерный или двумерный тензор, содержащий несколько переменных и наблюдений.
           Каждая строка тензора `m` представляет переменную, а каждый столбец — отдельное
           наблюдение для всех этих переменных.
        `rowvar`: Если `True`, то каждая строка представляет переменную, а наблюдения расположены
           в столбцах. Если `False`, то каждая колонка представляет переменную, а строки содержат наблюдения.

    Возвращает:
    -----------
        Матрицу ковариации переменных.
    """
    # Проверяем, что размерность входного тензора не превышает 2.
    if m.dim() > 2:
        raise ValueError('The covariance matrix can only be computed for an input with dimensionality no higher than 2')
    
    # Если тензор одномерный, преобразуем его в двумерный,
    # чтобы работать с ним как с матрицей (одна строка — одна переменная).
    if m.dim() < 2:
        m = m.view(1, -1)
    
    # Если rowvar=False, значит переменные представлены в столбцах.
    # При этом, если количество строк не равно 1, транспонируем тензор,
    # чтобы каждая строка соответствовала одной переменной.
    if not rowvar and m.size(0) != 1:
        m = m.t()
    
    # Вычисляем коэффициент нормировки: 1/(n-1),
    # где n — количество наблюдений (размер второго измерения после возможного транспонирования).
    fact = 1.0 / (m.size(1) - 1)
    
    # Центрируем данные: для каждой переменной вычитаем её среднее значение,
    # чтобы получить отклонения от среднего.
    m -= torch.mean(m, dim=1, keepdim=True)
    
    # Транспонируем центрированный тензор для удобства последующих матричных вычислений.
    # Если данные комплексные, нужно использовать сопряжённое транспонирование.
    if m.is_complex():
        mt = m.t().conj()
    else:
        mt = m.t() 
    
    # Вычисляем матрицу ковариации:
    # 1. Вычисляем произведение матрицы центрированных данных на её транспонированную версию,
    #    что приводит к сумме произведений отклонений для каждой пары переменных.
    # 2. Умножаем результат на коэффициент нормировки, чтобы получить несмещённую оценку ковариации.
    # 3. Метод squeeze() удаляет лишние размерности, если они возникли в результате вычислений.
    return fact * m.matmul(mt).squeeze()


def sqrt_newton_schulz(A: torch.Tensor, num_iterations: int, dtype=None) -> torch.Tensor:
    """
    Вычисляет матричный квадратный корень матрицы `A` методом Ньютона–Шульца.

    Данный метод использует итерационную схему для приближенного вычисления квадратного корня матрицы,
    путем последовательного обновления приближений `Y` и `Z`, где `Y` стремится к `A^(1/2)`, а `Z — к A^(-1/2)`.
    Для повышения численной стабильности входная матрица `A` нормализуется по её норме (по Фробениусу), после чего
    выполняется заданное число итераций. Итоговое приближение масштабируется обратно, возвращая квадратный корень `A`.

    Параметры:
    ----------
        `A` (Tensor): Батч квадратных матриц размера (batchSize, dim, dim), для которых необходимо вычислить квадратный корень.
        num_iterations (int): Количество итераций алгоритма. Увеличение числа итераций улучшает точность приближения, но увеличивает вычислительные затраты.
        dtype (torch.dtype, опционально): Тип данных для вычислений. Если не указан, используется тип данных матрицы `A`.

    Возвращает:
    -----------
        Tensor: Батч матриц, каждая из которых является приближением квадратного корня соответствующей матрицы `A`,
                т.е. для каждой матрицы `A` возвращается матрица `sA`, такая что `sA @ sA ≈ A`.

    Примечания:
    -----------
        - Алгоритм выполняется в контексте torch.no_grad(), поэтому вычисления не сохраняют градиенты.
        - Нормализация `A` и последующее масштабирование результата помогают обеспечить численную стабильность итераций.
    """
    # Отключаем вычисление градиентов, поскольку алгоритм используется для численного вычисления,
    # и нам не нужны градиенты для обратного распространения.
    with torch.no_grad():
        # Если тип данных не задан, используем тип данных входной матрицы A.
        if dtype is None:
            dtype = A.type()

    # Извлекаем размер батча (количество матриц) и размерность каждой квадратной матрицы.
    batch_size = A.shape[0]
    dim = A.shape[1]

    # Вычисляем норму каждой матрицы (по Фробениусу):
    # Для каждой матрицы в батче возводим все элементы в квадрат, суммируем по строкам и столбцам, 
    # затем берем квадратный корень от суммы.
    norm_of_A = A.mul(A).sum(dim=1).sum(dim=1).sqrt()

    # Нормализуем матрицы A: делим каждую матрицу на ее норму.
    # Это делается для повышения численной стабильности алгоритма.
    Y = A.div(norm_of_A.view(batch_size, 1, 1).expand_as(A))

    # Создаем единичную матрицу I размерности (dim x dim) и повторяем ее по батчу,
    # чтобы получить тензор формы (batch_size, dim, dim).
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).type(dtype)

    # Аналогично создаем матрицу Z, инициализированную как единичная матрица,
    # которая будет итеративно обновляться и стремиться к обратной матрице квадратного корня.
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).type(dtype)

    # Основной цикл итераций метода Ньютона–Шульца.
    for i in range(num_iterations):
        # Вычисляем вспомогательную матрицу T по формуле:
        # T = 0.5 * (3I - Z * Y)
        # Здесь используется батчевое матричное умножение (bmm).
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        
        # Обновляем матрицу Y: Y = Y * T.
        Y = Y.bmm(T)
        
        # Обновляем матрицу Z: Z = T * Z.
        Z = T.bmm(Z)

    # После завершения итераций получаем приближение матричного квадратного корня нормализованной матрицы.
    # Чтобы вернуться к масштабу исходной матрицы A, умножаем Y на sqrt(normA).
    sqrt_of_A = Y * torch.sqrt(norm_of_A).view(batch_size, 1, 1).expand_as(A)

    # Возвращаем приближенный квадратный корень исходной матрицы A.
    return sqrt_of_A
