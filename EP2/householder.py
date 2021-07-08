import numpy as np

class Householder:
  def __init__(self, A, print_steps=False):
    assert A.shape[0] == A.shape[1]

    self.n = A.shape[0]
    self.k = 0
    self.print_steps = print_steps

    self.A = np.ndarray(A.shape, dtype=float)

    self.Ht = np.identity(self.n, dtype=float)

    np.copyto(self.A, A)

  def _get_alfa_beta(self):
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
    if self.print_steps:
      print(f'Iteração {self.k+1}:')

    # Calcula o valor do wk da iteracao atual
    ak_ = self.A[self.k+1 :, [self.k]]

    wk_ = np.copy(ak_)
    wk_[0] += np.sign(ak_[0]) * np.linalg.norm(ak_)

    if self.print_steps:
      print(f'w{self.k+1}_:\n{wk_}')

    # Multiplica a matriz H_wk a esquerda de A
    self.A[self.k+1:, [self.k]] = ak_ - wk_

    for i in range(self.k+1, self.n):
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
    for m in range(self.n-2):
      self._iterate_once()

    result = {}
    result['T'] = self._get_alfa_beta()
    result['Ht'] = self.Ht

    return result



if __name__ == '__main__':
  A = np.array([[2, -1, 1, 3],
                [-1, 1, 4, 2],
                [1, 4, 2, -1],
                [3, 2, -1, 1]], dtype=float)

  hh = Householder(A, print_steps=False)

  # hh._iterate_once()
  result = hh.iterate()

  print(result['T']['alfa'])
  print(result['T']['beta'])
  print(result['Ht'])
