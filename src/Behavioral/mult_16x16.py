from argparse import ArgumentParser
import sys
#from mult_8x8_helper_functions import mult_4x4_approx
from Behavioral.mult_8x8 import mult_8x8_approx


def mult_16x16_approx(multip, mtc, adder_type):
    # Represent multiplier and multiplicand in 16-bits
    multip_bin = format(multip, '016b')
    mtc_bin = format(mtc, '016b')

    #Segment multiplier and multiplicand in 8-bit binary numbers.
    #Convert non-string to string and then integer
    multip_bin_lsb = int(str(multip_bin[8:16]), 2)  # note the indices
    multip_bin_msb = int(str(multip_bin[0:8]), 2)

    mtc_bin_lsb = int(str(mtc_bin[8:16]), 2)
    mtc_bin_msb = int(str(mtc_bin[0:8]), 2)

    product_LL = mult_8x8_approx(multip_bin_lsb, mtc_bin_lsb, adder_type)
    product_LH = mult_8x8_approx(multip_bin_lsb, mtc_bin_msb, adder_type)
    product_HL = mult_8x8_approx(multip_bin_msb, mtc_bin_lsb, adder_type)
    product_HH = mult_8x8_approx(multip_bin_msb, mtc_bin_msb, adder_type)

    if adder_type == 'acc':

        #Perfrom padding and shifting before addition
        product_LL_shift = int(str(product_LL).zfill(32))

        product_LH_shift = int(str(product_LH).zfill(24))
        product_LH_shift = product_LH_shift << 8

        product_HL_shift = int(str(product_HL).zfill(24))
        product_HL_shift = product_HL_shift << 8

        product_HH_shift = product_HH << 16

        product = product_LL_shift + product_LH_shift + product_HL_shift + product_HH_shift
        return (product)

    elif adder_type == 'app':

        # Do not add the N/2 LSBs and MSBs
        # Convert the result of individual 8x8 multipliers into binary

        product_LL_binary = format(product_LL, '016b')
        product_LH_binary = format(product_LH, '016b')
        product_HL_binary = format(product_HL, '016b')
        product_HH_binary = format(product_HH, '016b')

        # Separate the LSBs and MSBs before addition
        product_LL_binary_lsb = product_LL_binary[8:16]
        product_LL_binary_msb = product_LL_binary[0:8]

        product_LH_binary_lsb = product_LH_binary[8:16]
        product_LH_binary_msb = product_LH_binary[0:8]

        product_HL_binary_lsb = product_HL_binary[8:16]
        product_HL_binary_msb = product_HL_binary[0:8]

        product_HH_binary_lsb = product_HH_binary[8:16]
        product_HH_binary_msb = product_HH_binary[0:8]

        # add the eight MSBs of product_LL with eight LSBs of product_LH and product_HL
        # Does not propagate carries. Also discards carries e.g. 1 + 1 + 1 = 1
        # PP is for storing the result of this addition
        PP = [0] * 16
        for i in range(7, -1, -1):
            PP[i + 8] = (int(product_LL_binary_msb[i]) + int(
                product_LH_binary_lsb[i]) + int(product_HL_binary_lsb[i])) % 2

# Add the eight MSbs of product_LH and product_HL with eight LSBs of product_HH
        for j in range(7, -1, -1):
            PP[j] = (int(product_LH_binary_msb[j]) + int(
                product_HL_binary_msb[j]) + int(product_HH_binary_lsb[j])) % 2


# PP is a list. Convert list into string
        PP_str = ''.join(str(e) for e in PP)

        # Concatenate the four MSBs of product_HH with PP and four LSBs of product_LL
        product = product_HH_binary_msb + PP_str + product_LL_binary_lsb
        # Convert to integer
        product = int(product, 2)

        return (product)


def main():

    parser = ArgumentParser(
        description=
        'Implements 16x16 multiplier. 1st argument is multiplier and 2nd ardgument is multiplicand.'
    )
    parser.add_argument("multiplier", type=int, help="provide multiplier")
    parser.add_argument("multiplicand", type=int, help="provide multiplicand")
    parser.add_argument(
        "-add",
        "--adder",
        default='acc',
        type=str,
        help=
        'provide adder type: "acc" for accurate or "app" for approximate. Default is accurate'
    )

    args = parser.parse_args()

    #Convert numbers to integers
    multip = args.multiplier
    mtc = args.multiplicand
    adder_type = args.adder

    if multip > 65535 or mtc > 65535:
        print("Inputs out of bound. Exiting in 3 2 1 ...")
        sys.exit()

    product = mult_16x16_approx(multip, mtc, adder_type)
    print('product: ', product)


if __name__ == '__main__':
    main()
