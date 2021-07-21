import test_a
import test_b

HEADER = '''
#############################################
Aplicação do Algoritmo QR e de Transformações
de Householder

Jean Carlos Mello Xavier Faria - 11259628
Lucas Tonini Rosenberg Schneider - 11260850
#############################################
'''

DESC_A = 'Teste de cálculo de autovalores e autovetores de uma matriz real simétrica qualquer'
DESC_B = 'Cálculo das frequências e modos de vibrações de uma treliça plana'


def main():
  print(HEADER)
  print(f'Testes disponíveis:\n\tA - {DESC_A}\n\tB - {DESC_B}\n')
  test_letter = input('Digite a letra correspondente ao teste que deve ser executado: ')

  if test_letter == 'a' or test_letter == 'A':
    print('\n\n\n')
    test_a.run()
  elif test_letter == 'b' or test_letter == 'B':
    print('\n\n\n')
    test_b.run()
  else:
    print('Teste não encontrado, terminando...\n')

if __name__ == '__main__':
  main()
