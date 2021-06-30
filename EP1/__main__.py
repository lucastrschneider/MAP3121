from EP1 import tests

HEADER = '''
#############################################
Aplicação do Algoritmo QR

Jean Carlos Mello Xavier Faria - 11259628
Lucas Tonini Rosenberg Schneider - 11260850
#############################################
'''

DESC_A = 'Calculo de autovalores e autovetores para matriz A pre-definida'
DESC_B = 'Evolução da solução de um sistema massa-mola com 5 massas'
DESC_C = 'Evolução da solução de um sistema massa-mola com 10 massas'


def main():
  print(HEADER)
  print(f'Testes disponíveis:\n\tA - {DESC_A}\n\tB - {DESC_B}\n\tC - {DESC_C}\n')
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
