import numpy as np

class Householder:
  def __init__(self, A, print_steps=False):
    assert A.shape[0] == A.shape[1]

    self.n = A.shape[0]
    self.k = 0
    self.print_steps = print_steps

    self.A = np.ndarray(A.shape, dtype=float)

    np.copyto(self.A, A)


  def _iterate_once(self):

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

    

    self.k += 1

if __name__ == '__main__':
  A = np.array([[2, -1, 1, 3],
             [-1, 1, 4, 2],
             [1, 4, 2, -1],
             [3, 2, -1, 1]], dtype=float)

  hh = Householder(A, print_steps=True)

  hh._iterate_once()
