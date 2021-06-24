from EP1 import tests

def main():
  print('#############################################')
  print('Aplicação do Algoritmo QR\n')
  print('Jean Carlos Mello Xavier Faria - 11259628')
  print('Lucas Tonini Rosenberg Schneider - 11260850')
  print('#############################################\n')
  print('Testes disponíveis:\n\tA - Descrição\n\tB - Descrição\n\tC - Descrição\n')
  test_letter = input('Digite a letra correspondente ao teste que deve ser executado: ')

  if test_letter == 'a' or test_letter == 'A':
    print('\n\n\n')
    tests.test_a.run()
  elif test_letter == 'b' or test_letter == 'B':
    print('\n\n\n')
    tests.test_b.run()
  elif test_letter == 'c' or test_letter == 'C':
    print('\n\n\n')
    tests.test_c.run()
  else:
    print('Teste não encontrado, terminando...\n')

if __name__ == '__main__':
  main()
