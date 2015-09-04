# -*- coding: utf-8 -*-

# Python2.X encoding wrapper
import codecs,sys,numpy,pprint,re,random,pickle
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
from math import log10
from collections import defaultdict, Counter
import re, pprint

FLOAT_MIN = 2.2250738585072014
FLOAT_10EXP_MIN = -308.0
FLOAT_MAX = 1.7976931348623157
FLOAT_10EXP_MAX = 308.0



def main(alpha, beta):
	with codecs.open('result/test.txt', 'r', 'utf-8') as f:
		snt = [x.strip().split(u'//') for x in f.readlines()]

	pattern = re.compile(r'\{(.+?)\}')
	for s in snt:
		if not pattern.search(u''.join(s)):
			# print 'Delete Sentence:"%s"' %u' '.join(s)
			snt.remove(s)

	V = []		#全フレーム中の語彙の異なり総数(ex dobj:you)
	V_snt = []	#各フレームごとの語彙異なり語と頻度リスト	(ex [[{'dobj:you':25},{}..],[]...])
	C = defaultdict(list)		#各フレームｖと所属するクラスタCを対応付け
	# F = [{} for i in range(len(snt))]	#python2.6ではCounterがないので…

	for v in snt:
		snt_temp = {}
		for i, w in enumerate(v):
			try:
				if w[0] == u'<':
					temp = []
					temp.append(w.split(u'>:')[0][1:])
					temp_words = {}
					for s in pattern.search(w).group()[1:-1].split(u', '):
						s_split = s.split(u':')
						if not len(s_split[0]):
							continue
						temp_words[s_split[0]] = int(s_split[1])
						w_pair = u'%s:%s' %(temp[0], s_split[0])
						V.append(w_pair)
						snt_temp.update({w_pair: int(s_split[1])})
					temp.append(temp_words)
					v[i] = temp
			except IndexError:
				pass
		V_snt.append(snt_temp)

	V = list(set(V))
	# for i in range(len(snt)):
	# 	C[i] = []

	g_index = len(V_snt)

	"""
		ゴールドクラス読みだして、フレームと同じ形式で返す
	"""

	Gold = pickle_load(u'observe')
	gold = []
	for g in Gold['observe']:
		temp_word = {}
		gold_snts = []
		for w in g['parsed']:
			if w[1] == u'TARGET' and w[0] == u'observe':
				gold_snts.append(w[0])
			elif w[1] == u'root':
				pass
			else:
				gold_snts.append(u'%s:{%s:1}' %(w[1], w[0]))

			if w[1] != u'TARGET' and w[1] != u'root':
				w_pair = u'%s:%s' %(w[1], w[0])
				temp_word.update({w_pair : 1})
				V.append(w_pair)
		gold.append((temp_word, g['verbnet'], g['FN/PB']))
		snt.append(gold_snts)
		V_snt.append(temp_word)


	print 'INPUT PATTERNS:', len(V_snt)

	# print pp(gold)
	#verbnetとPropbankのタグを管理
	verbnet, propbank = defaultdict(list), defaultdict(list) 
	for i, g in enumerate(gold):
		verbnet[g[1]].append(i + g_index)
		propbank[g[2]].append(i + g_index)

	N = len(gold)

	#全初期フレームにランダムに所属するクラスタ番号を割り当てる
	for i in range(len(snt)):
		C[random.randint(0, len(snt)-1)].append(i)


	# alpha = 1	#αとβの値を決定
	# beta = 1
	F = [Counter() for i in range(len(snt))]


	for I in range(80):
		for i, n in enumerate(snt):
			for k, l in C.items():
				if i in l:			#viを所属クラスタC[k]から削除
					l.remove(i)
			# print 'snt[%d]'%i, C
			C_index = [k for k, l in C.items() if not l == []]
			new_i = random.choice(list(set(range(len(snt))) - set(C_index)))
			C_index.append(new_i)
			# print new_i, C_index
			P = {}
			for j in C_index:
				if j == new_i:
					P_w_f = beta - log10(len(V))*beta
					P_like = P_w_f * (len(n) - 1)
					P_prior = log10(alpha) - log10(len(snt) + alpha)
				else:
					P_like = 0
					for w in V:
						if w in V_snt[i]:
							# print w, j, C[j]
							count_f_w, count_f_t = 0, 0
							for snt_i in C[j]:
								if V_snt[snt_i].has_key(w):
									count_f_w += V_snt[snt_i][w]
								count_f_t += sum(V_snt[snt_i].values())
							# if count_f_w != 0:
							# print count_f_w, count_f_t
							log_Aw = log10(count_f_w + beta)
							log_Bw = log10(count_f_t + len(V)*beta)
							P_like += V_snt[i][w]*(log_Aw - log_Bw)

					P_prior = log10(len(C[j])) - log10(len(snt) + alpha)
				log_pst = P_prior + P_like
				P[j] = log_pst
				# if P[j] == 0.0:
					# P[j] = FLOAT_MIN
			# print pp(P)

			# plus_exp = -(min(P.values()) - FLOAT_10EXP_MIN)
			# plus_exp = - (sum(P.values()) / len(P))
			# print plus_exp
			p_max, p_min = max(P.values()), min(P.values())
			for index, p in P.items():
				# temp_logp = P[index] + plus_exp
				if p_min < FLOAT_10EXP_MIN:
					temp_logp = (p - p_min) / (p_max - p_min) - 1
					P[index] = 10**((temp_logp*(FLOAT_10EXP_MAX + p_max)) + p_max)
				else:
					P[index] = 10**p
				# print P[index]

			rand =  random.uniform(0, sum(P.values()))
			total = 0
			# index = 0
			for c_index, p in P.items():
				total += p
				if total > rand:
					# if i == 1:
					# print rand
					# print 'add Cluster -> %d' %c_index
					C[c_index].append(i)
					if I > 50:
						if F[i].has_key(c_index):
							F[i][c_index] += 1
						else:
							F[i][c_index] = 1
					break
				# index += 1

	Result_Class = defaultdict(list)
	for index, cnt in enumerate(F):
		# Result_Class[cnt.most_common(1)[0][0]].append('snt#%d' %index)
		Result_Class[cnt.most_common(1)[0][0]].append(index)

		# Result_Class[most_commom_indict(cnt)].append('snt#%d' %index)	#NICTマシン用

	print "number of class: " ,len(Result_Class)
	print Result_Class

	# print '\n\n'
	# print verbnet
	# print propbank
	print '\n\n'


	VNsum, PBsum = 0, 0
	for key in Result_Class:
		VNcounter, PBcounter = Counter(), Counter()
		for i in Result_Class[key]:
			for cls, v in verbnet.items():
				if i in v:
					VNcounter[cls] += 1
			for cls, v in propbank.items():
				if i in v:
					PBcounter[cls] += 1
		# print VNcounter, PBcounter

		try:
			#PUの値を計算
			VNsum += VNcounter.most_common(1)[0][1]
			PBsum += PBcounter.most_common(1)[0][1]

		except IndexError:
			pass


	#InversePUの値を計算
	VN_InPU_cnt, PB_InPU_cnt = 0, 0
	for cls, snt_list in verbnet.items():
		print cls,snt_list
		maxi = 0, 0		#recall, goldclass_num

		for key in Result_Class:
			goldclass_num = len(set(Result_Class[key]) & set(snt_list))
			content_num = len([i for i in Result_Class[key] if i >= g_index])
			recall = goldclass_num / float(len(snt_list))
			if recall > maxi[0]:
				maxi = recall, goldclass_num
		print maxi
		VN_InPU_cnt += maxi[1]
	
	for cls, snt_list in propbank.items():
		maxi = 0, 0 
		for key in Result_Class:
			goldclass_num = len(set(Result_Class[key]) & set(snt_list))
			content_num = len([i for i in Result_Class[key] if i >= g_index])
			recall = goldclass_num / float(len(snt_list))
			if recall > maxi[0]:
				maxi = recall, goldclass_num
		PB_InPU_cnt += maxi[1]


			# print VNcounter.most_common(1)[0], verbnet, Result_Class[key]
			# print len(verbnet[VNcounter.most_common(1)[0][0]])
			# print sum([x[1] for x in VNcounter.most_common()])
			# VN_InPU_cnt += (sum([x[1] for x in VNcounter.most_common()]) * VNcounter.most_common(1)[0][1]) / float(len(verbnet[VNcounter.most_common(1)[0][0]]))
			# PB_InPU_cnt += (sum([x[1] for x in PBcounter.most_common()]) * PBcounter.most_common(1)[0][1]) / float(len(propbank[PBcounter.most_common(1)[0][0]]))	


	COcounter = 0
	for i, g in enumerate(gold):
		gold_in_frame = sum([v for k, v in Result_Class.items() if i+g_index in v], [])
		if len(gold_in_frame) > 1:

			# print i+g_index, g
			
			gold_in_frame.remove(i+g_index)
			frame_snts = sum([V_snt[j].keys() for j in gold_in_frame], [])
			# print g[0], frame_snts
			if set(g[0].keys()) & set(frame_snts) == set(g[0].keys()):
				# print g[0], frame_snts
				COcounter += 1



	# print N
	print '*'*30
	print '\nVerbnet PU\t= %f' %(VNsum / float(N))
	print '\nPropBank PU\t= %f' %(PBsum / float(N))
	print '\nCoverage\t= %f' %(COcounter / float(len(gold)))
	print '\nVerbnet InPU\t= %f' %(VN_InPU_cnt / float(N))
	print '\nPropBank InPU\t= %f' %(PB_InPU_cnt / float(N))

	print '\nVerbnet  F-Score\t= %f' %(2 / ((N / float(VNsum)) + (N / float(VN_InPU_cnt)) ) )
	print '\nPropBank F-Score\t= %f' %(2 / ((N / float(PBsum)) + (N / float(PB_InPU_cnt)) ) )



def most_commom_indict(dic):
	return sorted(dic.iteritems(), key=lambda x: x[1], reverse=True)[0][0]

def pickle_load(name):
	with open('./goldclass/%s.pickle' %name, 'r') as f:
			return pickle.load(f)

def pp(obj):
	pp = pprint.PrettyPrinter(indent=4, width=160)
	str = pp.pformat(obj)
	return re.sub(r"\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1), 16)), str)

if __name__ == '__main__':
	main(int(sys.argv[1]), int(sys.argv[2]))
	