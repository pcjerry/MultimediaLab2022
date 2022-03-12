import os
from io import StringIO
import numpy as np
import huffman
import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time

# Jerry Xiong Multimedia labo 2022
# ========================= INFORMATION SOURCE =========================
# Information Source -> Source Encoding -> Channel Encoding -> Channel -> Channel Decoding -> Source Decoding -> Information Sink
print("============================================ INFORMATION SOURCE ============================================")
IMG_NAME = 'instagram.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
print(img)
# uncomment if you want to display the loaded image
# img.show()
# uncomment if you want to show the histogram of the colors
# image.show_color_hist()

# Use t.tic() and t.toc() to measure the executing time as shown below
t = Time()

img_pixelseq_uint8 = img.get_pixel_seq().copy()
img_bitseq_bit = util.uint8_to_bit(img_pixelseq_uint8)
print(img_pixelseq_uint8)

print("#Bits: " + str(len(img_bitseq_bit)))
print("====================================================================================================================================")
# ============================================ ENCODING ==============================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================
def huffman_encoder(bit_list):

    # Calculate huffman frequencies
    #huffman_freq = [(s, np.count_nonzero(img_pixelseg_uint8 == s)) for s in np.arange(256)]
    #huffman_tree = huffman.Tree(huffman_freq)

    imgHuffman_pixelseq_uint8 = util.bit_to_uint8(bit_list)
    np_pixelseq_uint8 = np.array(imgHuffman_pixelseq_uint8)
    (s, counts) = np.unique(np_pixelseq_uint8, return_counts=True)

    frequencies = np.asarray((s, counts)).T
    huffman_freq = frequencies

    t = Time()
    t.tic()
    huffman_tree = huffman.Tree(huffman_freq)
    print(F"Generating the Huffman Tree took {t.toc_str()}")
    print("Huffman Codebook: ")
    print(huffman_tree.codebook)

    t.tic()
    encoded_message = huffman.encode(huffman_tree.codebook, imgHuffman_pixelseq_uint8)
    print(F"Huffman Source Encoding took {t.toc_str()}")

    return huffman_tree, encoded_message


# Comment/uncomment Huffman and comment/uncomment LZW
print("============================================ SOURCE ENCODING HUFFMAN ============================================")
huffman_tree, source_encoded_bit = huffman_encoder(img_bitseq_bit)
print("#Bits: " + str(len(source_encoded_bit)))

# Padding (adding zeros): extra #bits to enable conversion to uint8 (divide by 8)
x = 8 - len(source_encoded_bit) % 8
print("Extra #bits required for conversion to uint8: " + str(x))

source_encoded_bit = source_encoded_bit.ljust(len(source_encoded_bit) + x, "0")
print("Huffman Source Encoding With Padding")
print("#Bits: " + str(len(source_encoded_bit)))
print("====================================================================================================================================")


# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
def lzw_encoder(input_lzw):
    t = Time()
    t.tic()

    encoded_message, dictionary = lzw.encode(input_lzw)
    print(F"LZW Source Encoding took {t.toc_str()}")

    encoded_bit = util.uintx_to_bit(encoded_message, width=16)

    return encoded_bit

'''print("============================================ SOURCE ENCODING LZW ============================================")
source_encoded_bit = lzw_encoder(img_bitseq_bit)
print("#Bits: " + str(len(source_encoded_bit)))
print("====================================================================================================================================")'''


# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
def reed_solomon_encoder(bit):
    # as we are working with symbols of 8 bits
    # choose n such that m is divisible by 8 when n=2^m−1
    # Example: 255 + 1 = 2^m -> m = 8
    n = 255  # code_word_length in symbols
    k = 223  # message_length in symbols

    coder = rs.RSCoder(n, k)

    imgRS_pixelseq_uint8 = util.bit_to_uint8(bit)

    x = k - imgRS_pixelseq_uint8.size % k

    size = imgRS_pixelseq_uint8.size
    symbols_padded = np.zeros((size + x,), dtype=int)
    symbols_padded[0:size] = imgRS_pixelseq_uint8

    messages = np.reshape(symbols_padded, (-1, k))
    rs_encoded_message = StringIO()

    t = Time()
    t.tic()
    for message in messages:
        code = coder.encode_fast(message, return_string=True)
        rs_encoded_message.write(code)

    rs_encoded_message_uint8 = np.array([ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

    print("RS Channel Encoding took: "+str(t.toc())+" s")
    print("Extra #Symbols: " + str(32 * symbols_padded.size / k))
    print("Extra #Symbols %: " + str(((rs_encoded_message_uint8.size - messages.size) / messages.size)*100) + " %")

    print("RS ENCODING COMPLETED")
    return util.uint8_to_bit(rs_encoded_message_uint8), messages

print("============================================ CHANNEL ENCODING REED_SOLOMON ============================================")


reed_solomon_bit, before_rs_encoded_message_uint8 = reed_solomon_encoder(source_encoded_bit)

print("RS Channel Encoding With Padding")
print("#Bits: " + str(len(reed_solomon_bit)))
print("====================================================================================================================================")


# =========================== CHANNEL ============================
# ================================================================
print("============================================ CHANNEL ============================================")

rs_encoded_message_bit = reed_solomon_bit

t.tic()
# Error rate value "ber" can be adjusted
print("Bit Errors introduced through channel")
received_rs_encoded_message_bit = channel(rs_encoded_message_bit, ber=0.10)
print(F"Bit Errors Implementation took {t.toc_str()}")
print("====================================================================================================================================")


# ====================== CHANNEL DECODING ========================
# ======================== Reed-Solomon ==========================
def reed_solomon_decoder(received_rs_encoded_message_bit, before_rs_encoded_message_uint8):

    received_rs_encoded_message_uint8 = util.bit_to_uint8(received_rs_encoded_message_bit)

    decoded_message = StringIO()

    t = Time()
    t.tic()

    n = 255  # code_word_length in symbols
    k = 223  # message_length in symbols

    coder = rs.RSCoder(n, k)

    # Only per 255 symbols decoding
    symbols = np.zeros(len(received_rs_encoded_message_uint8), dtype=int)
    symbols[0:len(received_rs_encoded_message_uint8)] = received_rs_encoded_message_uint8

    received_rs_encoded_message_uint8 = np.reshape(symbols, (-1, n))

    lend = 0
    for cnt, (block, original_block) in enumerate(zip(received_rs_encoded_message_uint8, before_rs_encoded_message_uint8)):
        try:
            decoded, ecc = coder.decode_fast(block, return_string=True, nostrip=True)
            assert coder.check(decoded + ecc), "Check not correct"
            decodeds = StringIO(decoded)
            lend = lend + len(np.array([ord(c) for c in decodeds.getvalue()], dtype=np.uint8))
            decoded_message.write(str(decoded))

        except rs.RSCodecError as error:
            decoded = StringIO(decoded)
            decoded = np.array([ord(c) for c in decoded.getvalue()], dtype=np.uint8)
            diff_symbols = len(decoded) - np.sum(original_block == decoded)
            print(F"Error occured after {cnt} iterations of {len(received_rs_encoded_message_uint8)}")
            print(F"{diff_symbols} different symbols in this block")

    rs_decoded_message_uint8 = np.array([ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)
    print("RS Channel Decoding took: " + str(t.toc()) + " s")
    print("#Symbols: " + str(len(rs_decoded_message_uint8)))

    return util.uint8_to_bit(rs_decoded_message_uint8)

print("============================================ CHANNEL DECODING REED_SOLOMON ============================================")
rs_decoded_message_bit = reed_solomon_decoder(received_rs_encoded_message_bit, before_rs_encoded_message_uint8)

# Remove padded symbols
rs_decoded_message_uint8 = util.bit_to_uint8(rs_decoded_message_bit)
print("Padded Symbols Removed")
rs_decoded_unpadded_message_uint8 = rs_decoded_message_uint8[0:len(util.bit_to_uint8(source_encoded_bit))]

print("RS DECODING COMPLETED")
print("Are information from sender and receiver after RS Channel Decoding identical?")
print(np.array_equal(util.bit_to_uint8(source_encoded_bit), rs_decoded_unpadded_message_uint8))
print("====================================================================================================================================")


# ======================= SOURCE DECODING ========================
# =========================== Huffman ============================
def huffman_decoder(huffman_tree, bit_list):
    t = Time()
    t.tic()
    decoded_message = huffman.decode(huffman_tree, bit_list)
    print(F"Huffman Source Decoding took {t.toc_str()}")

    img_bitseq_bit = util.uint8_to_bit(decoded_message)

    return img_bitseq_bit

# Comment/Uncomment Huffman or LZW
# Remove padded bits
print("============================================ SOURCE DECODING HUFFMAN ============================================")
rs_decoded_unpadded_message_bit = util.uint8_to_bit(rs_decoded_unpadded_message_uint8)

rs_decoded_unpadded_message_bit = rs_decoded_unpadded_message_bit[:-x]

img_bitseq_bit_received = huffman_decoder(huffman_tree, rs_decoded_unpadded_message_bit)

print('Huffman Source Decoding Without Padding')
print("#Bits: " + str(len(img_bitseq_bit_received)))
print("====================================================================================================================================")


# ======================= SOURCE DECODING ========================
# ====================== Lempel-Ziv-Welch ========================
def lzw_decoder(encoded_bit):
    encoded_message = util.bit_to_uint16(encoded_bit)

    t = Time()
    t.tic()
    #decoded_msg = lzw.decode(encoded_msg.tolist())
    #Convert to list
    decoded_message = lzw.decode(encoded_message.tolist())
    print(F"LZW Source Decoding took {t.toc_str()}")

    return decoded_message

# Remove padded bits
'''print("============================================ SOURCE DECODING LZW ============================================")
rs_decoded_unpadded_message_bit = util.uint8_to_bit(rs_decoded_unpadded_message_uint8)

img_bitseq_bit_received = lzw_decoder(rs_decoded_unpadded_message_bit)
#print(rs_decoded_unpadded_message_bit)
#print(img_bitseq_bit_received)

print('LZW Source Decoding Without Padding')
print("#Bits: " + str(len(img_bitseq_bit_received)))
print("====================================================================================================================================")'''

# =========================== SINK ===============================
# ================================================================
print("============================================ INFORMATION SINK ============================================")

# To prevent the usage of the original image here, empty everything before execution
img.img = None
img.bitmap = None
img.pixel_seq = None

#print(img_bitseq_bit_received)
# Huffman CHECK --- LZW ERROR
imgReceived_pixelseq_uint8 = util.bit_to_uint8(img_bitseq_bit_received)
print(imgReceived_pixelseq_uint8)

print("Received Bits -> (Pixelseq)Uint8 -> Bitmap -> Image")
print("Are information from sender and receiver after Huffman/LZW Source Decoding identical?")
print(np.array_equal(imgReceived_pixelseq_uint8, img_pixelseq_uint8))

imgReceived_bitmap = img.to_bitmap(imgReceived_pixelseq_uint8)
imgReceived = img.from_bitmap(imgReceived_bitmap)

print("Image show")
img.show()



