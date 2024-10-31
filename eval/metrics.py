from soynlp.hangle import decompose, compose

from typing import List
import re
from tqdm import tqdm

#import mecab
#mecab = mecab.MeCab()

class ASRMetrics:
    def __init__(self):
        pass
        
    def ser(self, ref, hyp ,debug=False, new=True):
        lines = []
        # split by phoneme
        r = list(ref)
        h = list(hyp)
        
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        DEL_PENALTY=1 # Tact
        INS_PENALTY=1 # Tact
        SUB_PENALTY=1 # Tact
        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r)+1):
            costs[i][0] = DEL_PENALTY*i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                    insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                    deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            # print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i-=1
                j-=1
                if debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub +=1
                i-=1
                j-=1
                if debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j-=1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i-=1
                if debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
                    
        if debug:
            reversed_lines = reversed(lines)
            for line in reversed_lines:
                print(line)
            print("Ncor " + str(numCor))
            print("Nsub " + str(numSub))
            print("Ndel " + str(numDel))
            print("Nins " + str(numIns))
            print("WER " + str((numSub + numDel + numIns) / (float) (len(r)))) 
            
        return numCor, numSub, numDel, numIns, (numSub + numDel + numIns) / (float) (len(r)), lines, (float) (len(r))


    def cer(self, ref, hyp ,debug=False, new=True):
        lines = []
        # split by phoneme
        ref = list(ref)
        r = []
        for c in ref:
            ctuple = decompose(c)
            if not ctuple:
                continue
            for ct in ctuple:
                if ct != ' ':
                    r.append(ct)
        hyp = list(hyp)
        h = []
        for c in hyp:
            ctuple = decompose(c)
            if not ctuple:
                continue
            for ct in ctuple:
                if ct != ' ':
                    h.append(ct)
        
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        DEL_PENALTY=1 # Tact
        INS_PENALTY=1 # Tact
        SUB_PENALTY=1 # Tact
        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r)+1):
            costs[i][0] = DEL_PENALTY*i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                    insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                    deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            # print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i-=1
                j-=1
                if debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub +=1
                i-=1
                j-=1
                if debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j-=1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i-=1
                if debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
                    
        if debug:
            reversed_lines = reversed(lines)
            for line in reversed_lines:
                print(line)
            print("Ncor " + str(numCor))
            print("Nsub " + str(numSub))
            print("Ndel " + str(numDel))
            print("Nins " + str(numIns))
            print("WER " + str((numSub + numDel + numIns) / (float) (len(r)))) 
        return numCor, numSub, numDel, numIns, (numSub + numDel + numIns) / (float) (len(r)), lines, (float) (len(r))

    # Helper functions
    def clean_text(self, text: str) -> str:
        text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text.replace('\n', '')).strip()
        return text

    def remove_sound(self, text: str) -> str:
        results = []
        for word in text.split():
            if len(word) == 1:
                continue
            results.append(word)
        return ' '.join(results)
    
    def ignore_similar(self, text: str) -> str:
        results = []
        for c in text:
            # Try decomposing the character
            phonemes = decompose(c)
            if phonemes is None :
                # If decomposition fails, keep the character as is
                results.append(c)
                continue
            onset, nucleus, coda = phonemes

            if nucleus in ['ㅔ', 'ㅐ', 'ㅒ', 'ㅖ']:
                nucleus = 'ㅔ'
            elif nucleus in  ['ㅙ', 'ㅚ', 'ㅞ']:
                nucleus = 'ㅙ'
            results.append(compose(onset, nucleus, coda))
        return ''.join(results)

    def preprocess_text(self, text: str, debug: bool = False) -> str:
        text = self.clean_text(text)
        if debug:
            print(f"Clean text: {text}")
        # Remove sound
        #text = self.remove_sound(text)
        if debug:
            print(f"Remove sound: {text}")
        # Ignore similar phonemes
        text = self.ignore_similar(text)
        if debug:
            print(f"Ignore similar phonemes: {text}")
        return text
    
    # Main pipeline
    def calculate_metrics_averaged(self, all_predictions: List[str], all_references: List[str]) -> None:
        total_ser = 0
        total_cer = 0

        total_ins1 = 0
        total_del1 = 0
        total_sub1 = 0

        total_ins2 = 0
        total_del2 = 0
        total_sub2 = 0

        total_len = len(all_references)

        for r, p in tqdm(zip(all_references, all_predictions)):
            r = self.preprocess_text(r, debug=False)
            p = self.preprocess_text(p, debug=False)
            if not r or not p:
                continue
            numCor1, numSub1, numDel1, numIns1, ser_score, lines, total1 = self.ser(r, p, debug=False)
            numCor2, numSub2, numDel2, numIns2, cer_score, lines, total2 = self.cer(r, p, debug=False)
            #dist, length, cer_score = cer2(r, p)
            total_ser += ser_score
            total_cer += cer_score

            total_ins1 += numIns1/total1
            total_del1 += numDel1/total1
            total_sub1 += numSub1/total1

            total_ins2 += numIns2/total2
            total_del2 += numDel2/total2
            total_sub2 += numSub2/total2

        total_ser /= total_len
        total_cer /= total_len

        total_ins1 /= total_len
        total_del1 /= total_len
        total_sub1 /= total_len

        total_ins2 /= total_len
        total_del2 /= total_len
        total_sub2 /= total_len

        print(f"SER: {total_ser*100:.2f}%")
        print(f"INS SER: {total_ins1*100:.2f}%")
        print(f"DEL SER: {total_del1*100:.2f}%")
        print(f"SUB SER: {total_sub1*100:.2f}%")
        print('-'*13)
        print(f"CER: {total_cer*100:.2f}%")
        print(f"INS CER: {total_ins2*100:.2f}%")
        print(f"DEL CER: {total_del2*100:.2f}%")
        print(f"SUB CER: {total_sub2*100:.2f}%")

    def calculate_metrics_batched(self, all_predictions: List[str], all_references: List[str]) -> None:
        total_ser_numerator = 0  # Total number of substitutions, deletions, and insertions for SER
        total_cer_numerator = 0  # Total number of substitutions, deletions, and insertions for CER

        total_ins1, total_del1, total_sub1 = 0, 0, 0  # SER components
        total_ins2, total_del2, total_sub2 = 0, 0, 0  # CER components

        total_words = 0  # Total words for SER normalization
        total_chars = 0  # Total characters for CER normalization

        # Process each prediction and reference pair
        for r, p in tqdm(zip(all_references, all_predictions), total=len(all_references)):
            try:
                r = self.preprocess_text(r, debug=False)
                p = self.preprocess_text(p, debug=False)
            except Exception as e:
                print(f"{e} occurred for {r} and {p}")
                continue
            if not r or not p:
                continue

            # SER metrics
            try:
                numCor1, numSub1, numDel1, numIns1, ser_score, lines, total1 = self.ser(r, p, debug=False)
                total_words += total1
                total_sub1 += numSub1
                total_del1 += numDel1
                total_ins1 += numIns1
                total_ser_numerator += numSub1 + numDel1 + numIns1

                # CER metrics
                numCor2, numSub2, numDel2, numIns2, cer_score, lines, total2 = self.cer(r, p, debug=False)
                total_chars += total2
                total_sub2 += numSub2
                total_del2 += numDel2
                total_ins2 += numIns2
                total_cer_numerator += numSub2 + numDel2 + numIns2
            except Exception as e:
                print(f"{e} occurred for {r} and {p}")
                continue

        # Final SER and CER calculations (batch-based)
        total_ser = total_ser_numerator / total_words if total_words > 0 else 0
        total_cer = total_cer_numerator / total_chars if total_chars > 0 else 0

        # Normalize the individual components
        ins_ser = total_ins1 / total_words if total_words > 0 else 0
        del_ser = total_del1 / total_words if total_words > 0 else 0
        sub_ser = total_sub1 / total_words if total_words > 0 else 0

        ins_cer = total_ins2 / total_chars if total_chars > 0 else 0
        del_cer = total_del2 / total_chars if total_chars > 0 else 0
        sub_cer = total_sub2 / total_chars if total_chars > 0 else 0

        # Print the final results
        print(f"SER: {total_ser * 100:.2f}%")
        print(f"INS SER: {ins_ser * 100:.2f}%")
        print(f"DEL SER: {del_ser * 100:.2f}%")
        print(f"SUB SER: {sub_ser * 100:.2f}%")
        print('-' * 13)
        print(f"CER: {total_cer * 100:.2f}%")
        print(f"INS CER: {ins_cer * 100:.2f}%")
        print(f"DEL CER: {del_cer * 100:.2f}%")
        print(f"SUB CER: {sub_cer * 100:.2f}%")
        return total_ser, total_cer



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    speaker = 'JHJ_Woman_40s-01-08-35.48'
    parser.add_argument("--predictions", type=str, default=f'stuff/predictions_VC-20dB-{speaker}_valid.txt')
    parser.add_argument("--references", type=str, default=f'stuff/references_VC-20dB-{speaker}_valid.txt')
    args = parser.parse_args()

    # test
    with open(args.predictions, 'r') as f:
        ref = f.readlines()
    with open(args.references, 'r') as f:
        pred = f.readlines()

    metrics = ASRMetrics()
    metrics.calculate_metrics_batched(pred, ref)