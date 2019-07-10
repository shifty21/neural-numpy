from argparse import ArgumentParser
import sys
def mult_4x2 (multip, multic):
  print('this is a function', multip, multic)
  prod = multic * multip
  if prod % 2 != 0:
  	prod = prod - 1
  	  
  return(prod)


def main():
	parser = ArgumentParser(description='Implements 4x2 multiplier. 1st argument is multiplicand and 2nd ardgument is multiplier.')
	parser.add_argument("multiplier", type = int, help="provide multiplier")
	parser.add_argument("multiplicand", type = int, help="provide multiplicand")

	args = parser.parse_args()

#Extract Operands	
	multip = args.multiplier
	multic = args.multiplicand
	if multip > 3 or multic > 15:
		print('\n This is a 4x2 multiplier')
		print('Operands out of bound.... exiting')
		sys.exit()

	product = mult_4x2(multip, multic)
	print ('product: ', product)

if __name__ == '__main__':
	main()