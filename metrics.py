from soynlp.hangle import decompose, compose

from typing import List
import re
from tqdm import tqdm

import mecab


class ASRMetrics:
    def __init__(self):
        self.mecab = mecab.MeCab()

    def wer(self, ref, hyp, debug=False, new=True, split='pos'): 
        """_summary_

        Args:
            ref (string): reference sentence
            hyp (string): prediction sentence
            debug (bool, optional): Show details. Defaults to False.
            split (str, optional): sentence split strategy. Defaults to 'whitespace'.

        Returns:numCor, numSub, numDel, numIns, len(r), (numSub + numDel + numIns) / (float) (len(r)), lines
            tuple: number of Correction, number of substitutions, number of deletions, number of insertion, WER score, detail list
        """
        lines = []
        assert split in ['whitespace', 'pos']
        
        if split == 'whitespace':
            # 띄어쓰기 기반 어절분리
            r = ref.split()
            h = hyp.split()
        
        elif split == 'pos':
            # 형태소기반 어절분리
            r = self.mecab.morphs(ref)
            h = self.mecab.morphs(hyp)
        
        # # 토큰 기반 어절분리
        # r = [x.replace('▁', '') for x in tokenizer.tokenize(ref)]
        # h = [x.replace('▁', '') for x in tokenizer.tokenize(hyp)]
        
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
            # for line in reversed_lines:
            #     print(line)
            # print("Ncor " + str(numCor))
            # print("Nsub " + str(numSub))
            # print("Ndel " + str(numDel))
            # print("Nins " + str(numIns))
            # print("WER " + str((numSub + numDel + numIns) / (float) (len(r)))) 
        score = (numSub + numDel + numIns) / (float) (len(r)) if len(r) > 0 else 0
        return numCor, numSub, numDel, numIns, score, lines, (float) (len(r))
                
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
        score = (numSub + numDel + numIns) / (float) (len(r)) if len(r) > 0 else 0
        return numCor, numSub, numDel, numIns, score, lines, (float) (len(r))
    
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
        score = (numSub + numDel + numIns) / (float) (len(r)) if len(r) > 0 else 0
        return numCor, numSub, numDel, numIns, score, lines, (float) (len(r))

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
        total_wer = 0
        total_wer_no_ins = 0

        total_ins1 = 0
        total_del1 = 0
        total_sub1 = 0

        total_ins2 = 0
        total_del2 = 0
        total_sub2 = 0

        total_ins3 = 0
        total_del3 = 0
        total_sub3 = 0

        total_len = len(all_references)

        for r, p in tqdm(zip(all_references, all_predictions)):
            r = self.preprocess_text(r, debug=False)
            p = self.preprocess_text(p, debug=False)
            if not r or not p:
                continue
            numCor1, numSub1, numDel1, numIns1, ser_score, lines, total1 = self.ser(r, p, debug=False)
            numCor2, numSub2, numDel2, numIns2, cer_score, lines, total2 = self.cer(r, p, debug=False)
            numCor3, numSub3, numDel3, numIns3, wer_score, lines, total3 = self.wer(r, p, debug=False)
            
            #dist, length, cer_score = cer2(r, p)
            total_ser += ser_score
            total_cer += cer_score
            total_wer += wer_score
            total_wer_no_ins += (numSub3 + numDel3) / total3 if total3 > 0 else 0

            total_ins1 += numIns1/total1
            total_del1 += numDel1/total1
            total_sub1 += numSub1/total1

            total_ins2 += numIns2/total2
            total_del2 += numDel2/total2
            total_sub2 += numSub2/total2

            total_ins3 += numIns3/total3
            total_del3 += numDel3/total3
            total_sub3 += numSub3/total3

        total_ser /= total_len
        total_cer /= total_len
        total_wer /= total_len
        total_wer_no_ins /= total_len

        total_ins1 /= total_len
        total_del1 /= total_len
        total_sub1 /= total_len

        total_ins2 /= total_len
        total_del2 /= total_len
        total_sub2 /= total_len

        total_ins3 /= total_len
        total_del3 /= total_len
        total_sub3 /= total_len

        print(f"SER: {total_ser*100:.2f}%")
        print(f"INS SER: {total_ins1*100:.2f}%")
        print(f"DEL SER: {total_del1*100:.2f}%")
        print(f"SUB SER: {total_sub1*100:.2f}%")
        print('-'*13)
        print(f"CER: {total_cer*100:.2f}%")
        print(f"INS CER: {total_ins2*100:.2f}%")
        print(f"DEL CER: {total_del2*100:.2f}%")
        print(f"SUB CER: {total_sub2*100:.2f}%")
        print('-'*13)
        print(f"WER: {total_wer*100:.2f}%")
        print(f"WER (no INS): {total_wer_no_ins*100:.2f}%")
        print(f"INS WER: {total_ins3*100:.2f}%")
        print(f"DEL WER: {total_del3*100:.2f}%")
        print(f"SUB WER: {total_sub3*100:.2f}%")
        return total_ser, total_cer, total_wer, total_wer_no_ins

    def calculate_metrics_batched(self, all_predictions: List[str], all_references: List[str]) -> None:
        total_ser_numerator = 0  # Total number of substitutions, deletions, and insertions for SER
        total_cer_numerator = 0  # Total number of substitutions, deletions, and insertions for CER
        total_wer_numerator = 0  # Total number of substitutions, deletions, and insertions for WER
        total_wer_no_ins_numerator = 0  # Total number of substitutions and deletions for WER

        total_ins1, total_del1, total_sub1 = 0, 0, 0  # SER components
        total_ins2, total_del2, total_sub2 = 0, 0, 0  # CER components
        total_ins3, total_del3, total_sub3 = 0, 0, 0  # WER components

        total_sylls = 0  # Total words for SER normalization
        total_chars = 0  # Total characters for CER normalization
        total_words = 0  # Total words for WER normalization


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
                total_sylls += total1
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

                # WER metrics
                numCor3, numSub3, numDel3, numIns3, wer_score, lines, total3 = self.wer(r, p, debug=False)
                total_words += total3
                total_sub3 += numSub3
                total_del3 += numDel3
                total_ins3 += numIns3
                total_wer_numerator += numSub3 + numDel3 + numIns3
                total_wer_no_ins_numerator += numSub3 + numDel3

            except Exception as e:
                print(f"{e} occurred for {r} and {p}")
                continue

        # Final SER and CER calculations (batch-based)
        total_ser = total_ser_numerator / total_sylls if total_sylls > 0 else 0
        total_cer = total_cer_numerator / total_chars if total_chars > 0 else 0
        total_wer = total_wer_numerator / total_words if total_words > 0 else 0
        total_wer_no_ins = total_wer_no_ins_numerator / total_words if total_words > 0 else 0

        # Normalize the individual components
        ins_ser = total_ins1 / total_sylls if total_sylls > 0 else 0
        del_ser = total_del1 / total_sylls if total_sylls > 0 else 0
        sub_ser = total_sub1 / total_sylls if total_sylls > 0 else 0

        ins_cer = total_ins2 / total_chars if total_chars > 0 else 0
        del_cer = total_del2 / total_chars if total_chars > 0 else 0
        sub_cer = total_sub2 / total_chars if total_chars > 0 else 0

        ins_wer = total_ins3 / total_words if total_words > 0 else 0
        del_wer = total_del3 / total_words if total_words > 0 else 0
        sub_wer = total_sub3 / total_words if total_words > 0 else 0

        # Print the final results  
        print(f"WER: {total_wer * 100:.2f}%")
        print(f"WER (no INS): {total_wer_no_ins * 100:.2f}%")
        print(f"INS WER: {ins_wer * 100:.2f}%")
        print(f"DEL WER: {del_wer * 100:.2f}%")
        print(f"SUB WER: {sub_wer * 100:.2f}%")
        print('-' * 13)
        print(f"SER: {total_ser * 100:.2f}%")
        print(f"INS SER: {ins_ser * 100:.2f}%")
        print(f"DEL SER: {del_ser * 100:.2f}%")
        print(f"SUB SER: {sub_ser * 100:.2f}%")
        print('-' * 13)
        print(f"CER: {total_cer * 100:.2f}%")
        print(f"INS CER: {ins_cer * 100:.2f}%")
        print(f"DEL CER: {del_cer * 100:.2f}%")
        print(f"SUB CER: {sub_cer * 100:.2f}%")


        return total_ser, total_cer, total_wer_numerator, total_wer_no_ins_numerator



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    speaker = '13_CUJ_Woman-00-31-31.64'
    parser.add_argument("--predictions", type=str, default=f'stuff/predictions_VC-20dB-lora-{speaker}_valid.txt')
    parser.add_argument("--references", type=str, default=f'stuff/references_VC-20dB-lora-{speaker}_valid.txt')
    args = parser.parse_args()
    
    metrics = ASRMetrics()

    print(f"Calculating metrics for {args.predictions} and {args.references}")

    # valid
    print("Validation set:")
    with open(args.predictions, 'r') as f:
        ref = f.readlines()
    with open(args.references, 'r') as f:
        pred = f.readlines()
    metrics.calculate_metrics_batched(pred, ref)

    # test
    print("\nTest set:")
    with open(args.predictions.replace('valid', 'test'), 'r') as f:
        ref = f.readlines()
    with open(args.references.replace('valid', 'test'), 'r') as f:
        pred = f.readlines()
    metrics.calculate_metrics_batched(pred, ref)