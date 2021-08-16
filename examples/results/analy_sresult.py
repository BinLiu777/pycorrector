from collections import Counter
import math


with open('order_pure_cutword_nopoun_stopwords_texts_model1.res', 'r') as f:
    lines = f.readlines()
    ppls = []
    logppls = []
    for line in lines:
        if 'ppl' in line:
            # logprob = float(line.strip().split()[0][8:])
            ppl = float(line.strip().split()[1][4:])
            ppls.append(int(ppl/100))
            logppls.append(round(math.log10(ppl),1))
    counter = Counter()
    counter.update(ppls)
    print(counter)
    counter = Counter()
    counter.update(logppls)
    print(counter)
with open('noorder_pure_cutword_nopoun_stopwords_texts_model1.res', 'r') as f:
    lines = f.readlines()
    ppls = []
    logppls = []
    for line in lines:
        if 'ppl' in line:
            # logprob = float(line.strip().split()[0][8:])
            ppl = float(line.strip().split()[1][4:])
            ppls.append(int(ppl/100))
            logppls.append(round(math.log10(ppl),1))
    counter = Counter()
    counter.update(ppls)
    print(counter)
    counter = Counter()
    counter.update(logppls)
    print(counter)
# with open('order_pure_cutword_nopoun_stopwords_texts_model2.res', 'r') as f:
#     lines = f.readlines()
#     ppls = []
#     for line in lines:
#         if 'ppl' in line:
#             # logprob = float(line.strip().split()[0][8:])
#             ppl = float(line.strip().split()[1][4:])
#             ppls.append(int(ppl/100))
#     counter = Counter()
#     counter.update(ppls)
#     print(counter)
# with open('noorder_pure_cutword_nopoun_stopwords_texts_model2.res', 'r') as f:
#     lines = f.readlines()
#     ppls = []
#     for line in lines:
#         if 'ppl' in line:
#             # logprob = float(line.strip().split()[0][8:])
#             ppl = float(line.strip().split()[1][4:])
#             ppls.append(int(ppl/100))
#     counter = Counter()
#     counter.update(ppls)
#     print(counter)

Counter({1: 60711, 0: 45722, 2: 27057, 3: 14477, 4: 8595, 5: 5730, 6: 4332, 7: 3553, 10: 3477, 8: 2666, 14: 2320, 12: 2131, 9: 1527, 11: 908, 13: 739, 37: 685, 19: 641, 20: 473, 43: 471, 15: 471, 16: 463, 17: 353, 18: 294, 88: 259, 21: 211, 39: 203, 29: 198, 23: 196, 118: 185, 24: 176, 28: 162, 40: 160, 31: 160, 22: 160, 25: 150, 98: 142, 26: 133, 128: 114, 80: 100, 57: 100, 30: 98, 27: 84, 35: 81, 170: 79, 34: 75, 44: 74, 33: 73, 32: 73, 45: 65, 38: 65, 166: 60, 41: 44, 46: 41, 69: 39, 36: 39, 47: 37, 48: 36, 42: 34, 54: 34, 86: 28, 188: 27, 49: 27, 58: 26, 52: 25, 85: 24, 61: 21, 109: 21, 146: 20, 60: 19, 92: 18, 81: 15, 56: 15, 72: 14, 51: 14, 63: 14, 55: 14, 65: 13, 50: 12, 99: 12, 64: 10, 77: 9, 59: 9, 62: 9, 53: 8, 186: 8, 74: 7, 76: 7, 89: 6, 78: 6, 337: 5, 67: 5, 71: 5, 388: 5, 70: 5, 104: 5, 116: 5, 82: 4, 100: 4, 84: 4, 196: 4, 68: 4, 127: 4, 3601: 4, 66: 4, 106: 3, 83: 3, 162: 3, 152: 3, 110: 3, 107: 3, 215: 3, 332: 2, 108: 2, 117: 2, 105: 2, 95: 2, 374: 2, 102: 2, 93: 2, 133: 2, 338: 2, 96: 2, 497: 2, 136: 2, 142: 2, 119: 1, 352: 1, 168: 1, 91: 1, 198: 1, 103: 1, 240: 1, 79: 1, 123: 1, 144: 1, 481: 1, 113: 1, 176: 1, 139: 1, 73: 1, 130: 1, 87: 1, 126: 1, 94: 1, 101: 1, 165: 1, 112: 1})
Counter({1: 22644, 2: 13131, 0: 12475, 3: 7946, 4: 6140, 5: 4368, 6: 2743, 7: 2303, 8: 1706, 10: 1455, 9: 1097, 14: 942, 12: 667, 11: 640, 86: 628, 13: 594, 88: 575, 20: 406, 43: 374, 16: 362, 15: 355, 31: 351, 17: 294, 19: 283, 37: 265, 170: 252, 18: 200, 21: 180, 24: 161, 22: 146, 23: 133, 39: 131, 128: 130, 28: 116, 3601: 114, 57: 112, 25: 112, 26: 100, 188: 94, 29: 94, 27: 82, 38: 77, 126: 75, 40: 74, 32: 71, 30: 69, 33: 68, 54: 68, 47: 68, 44: 66, 45: 60, 166: 53, 34: 52, 35: 40, 48: 39, 81: 36, 53: 35, 42: 33, 374: 32, 41: 29, 49: 29, 52: 25, 60: 25, 85: 25, 36: 24, 118: 23, 51: 22, 46: 20, 98: 19, 61: 17, 58: 16, 50: 16, 146: 15, 123: 14, 64: 14, 71: 14, 55: 14, 56: 13, 77: 13, 65: 11, 109: 11, 59: 11, 80: 11, 75: 11, 185: 10, 95: 10, 62: 10, 337: 9, 97: 9, 69: 9, 127: 9, 101: 8, 130: 8, 99: 7, 92: 7, 84: 7, 79: 6, 70: 6, 103: 6, 90: 6, 63: 6, 107: 6, 82: 5, 67: 5, 89: 5, 114: 5, 68: 5, 72: 4, 74: 4, 66: 4, 196: 4, 76: 4, 117: 4, 106: 3, 563: 3, 73: 3, 201: 3, 104: 3, 91: 3, 134: 3, 140: 3, 243: 3, 78: 3, 219: 2, 271: 2, 113: 2, 291: 2, 100: 2, 87: 2, 160: 2, 96: 2, 246: 2, 125: 2, 115: 2, 93: 2, 102: 2, 145: 1, 250: 1, 149: 1, 120: 1, 205: 1, 111: 1, 425: 1, 237: 1, 957: 1, 142: 1, 83: 1, 242: 1, 289: 1, 191: 1, 203: 1, 151: 1, 189: 1, 182: 1, 195: 1, 192: 1, 413: 1, 163: 1, 190: 1, 274: 1, 249: 1, 224: 1, 121: 1, 240: 1, 133: 1, 94: 1, 147: 1, 159: 1, 230: 1, 253: 1, 179: 1, 135: 1, 239: 1, 183: 1, 155: 1, 112: 1, 231: 1, 264: 1, 153: 1, 186: 1, 206: 1, 305: 1})
