import numpy as np

class Householder:
  """
  Classe que representa o algoritmo utilizado para fazer transformações de Householder
  para reduzir uma matriz simétrica a uma tridiagonal simetrica semelhante.

  Args:
      A (numpy.ndarray): Array n por n que representa a matriz completa a ser reduzida.
      print_steps (bool, optional): Indica se deve ou não imprimir os passos intermediários. Padrão é False.

  Attributes:
    n (int): Variável que indica a dimensão da matriz do sistema.
    k (int): Guarda o número da iteração atual do algorítmo.
    print_steps (bool): Indica se deve imprimir informações de cada iteração na tela.
    A (numpy.ndarray): Guarda o valor atual da matriz A.
    Ht (numpy.ndarray): Guarda o valor atual da matriz H transposta.
  """
  def __init__(self, A, print_steps=False):
    assert A.shape[0] == A.shape[1]

    self.n = A.shape[0]
    self.k = 0
    self.print_steps = print_steps

    self.A = np.ndarray(A.shape, dtype=float)

    self.Ht = np.identity(self.n, dtype=float)

    np.copyto(self.A, A)

  def _get_alfa_beta(self):
    """
    Retorna a diagonal pricipal e subdiagonal da matriz tridiagonalizada pelo algoritmo.

    Returns:
        dict: Dicionário com dois elementos, um numpy.ndarray representando a diagonal principal
                cuja chave é "alfa", e um representando as subdiagonais, com chave "beta".
    """
    alfa = np.ndarray(self.n, dtype=float)
    beta = np.ndarray(self.n-1, dtype=float)

    for i in range(self.n-1):
      alfa[i] = self.A[i][i]
      beta[i] = self.A[i+1][i]

    alfa[self.n-1] = self.A[self.n-1][self.n-1]

    result = {
      'alfa' : alfa,
      'beta' : beta,
    }

    return result

  def _iterate_once(self):
    """
    Método que representa uma iteração de transformações de Householder para uma matriz simétrica.
    """
    if self.print_steps:
      print(f'Iteração {self.k+1}:')

    # Calcula o valor do wk da iteracao atual
    ak_ = self.A[self.k+1 :, [self.k]]

    wk_ = np.copy(ak_)
    wk_[0] += np.sign(ak_[0]) * np.linalg.norm(ak_)

    if self.print_steps:
      print(f'w{self.k+1}_:\n{wk_}')

    # Multiplica a matriz H_wk a esquerda de A
    for i in range(self.k, self.n):
      ai_ = self.A[self.k+1 :, [i]]

      self.A[self.k+1 :, [i]] = ai_ - 2 * np.dot(ai_.T, wk_) / np.dot(wk_.T, wk_) * wk_

    if self.print_steps:
      print(f'H_w{self.k+1} A:\n{self.A}')

    # Multiplica a matriz H_wk a direita de A
    for i in range(self.k+1, self.n):
      bi_ = self.A[[i], self.k+1 :]

      self.A[[i], self.k+1 :] = bi_ - 2 * np.dot(bi_, wk_) / np.dot(wk_.T, wk_) * wk_.T

    if self.print_steps:
      print(f'H_w{self.k+1} A H_w{self.k+1}:\n{self.A}')

    # Multiplica a matriz H_wk a direita de Ht
    for i in range(0, self.n):
      bi_ = self.Ht[[i], self.k+1 :]

      self.Ht[[i], self.k+1 :] = bi_ - 2 * np.dot(bi_, wk_) / np.dot(wk_.T, wk_) * wk_.T

    if self.print_steps:
      print(f'Ht H_w{self.k+1}:\n{self.Ht}\n')

    self.k += 1

  def iterate(self):
    """
    Método que executa as transformações de Householder para a matriz fornecida no construtor.

    Returns:
        dict: Dicionário contendo a matriz tridiagonal e a matriz tranposta de H, para serem
                utilizados pelo método QR.
    """
    for m in range(self.n-2):
      self._iterate_once()

    result = {}
    result['T'] = self._get_alfa_beta()
    result['Ht'] = self.Ht

    return result



if __name__ == '__main__':
  print('Esse arquivo contem apenas o algoritmo Householder, mas não os testes.')
  print('Para rodar os testes corretamente, por favor consulte o LEIAME.txt')
