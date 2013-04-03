echo '1 done'
python sort.py ./ar_un/arabic_1_ortho f1
python sort.py ./ar_un/arabic_1_conxt f2
python combine.py f1 f2  ./ar_un/arabic_1_all
echo '2 done'
python sort.py ./ar_un/arabic_2_ortho f1
python sort.py ./ar_un/arabic_2_conxt f2
python combine.py f1 f2  ./ar_un/arabic_2_all
echo '3 done'
sort ./ar_un/english_1_ortho > f1
sort ./ar_un/english_1_conxt > f2
python combine.py f1 f2  ./ar_un/english_1_all
echo '4 done'
python sort.py ./ch_g/chinese_g_ortho f1
python sort.py ./ch_g/chinese_g_conxt f2
python combine.py f1 f2  ./ch_g/chinese_g_all
echo '5 done'
python sort.py ./ch_un/chinese_1_ortho f1
python sort.py ./ch_un/chinese_1_conxt f2
python combine.py f1 f2  ./ch_un/chinese_1_all
echo '6 done'
python sort.py ./ch_un/chinese_2_ortho f1
python sort.py ./ch_un/chinese_2_conxt f2
python combine.py f1 f2  ./ch_un/chinese_2_all
echo '7 done'
sort ./ch_un/english_1_ortho > f1
sort ./ch_un/english_1_conxt > f2
python combine.py f1 f2  ./ch_un/english_1_all
echo '8 done'
sort ./en_g/english_g_ortho > f1
sort ./en_g/english_g_conxt > f2
python combine.py f1 f2  ./en_g/english_g_all
echo '9 done'
sort ./fr_eu/french_1_ortho > f1
sort ./fr_eu/french_1_conxt > f2
python combine.py f1 f2  ./fr_eu/french_1_all
echo '10 done'
sort ./fr_eu/french_2_ortho > f1
sort ./fr_eu/french_2_conxt > f2
python combine.py f1 f2  ./fr_eu/french_2_all
echo '11 done'
sort ./fr_eu/english_1_ortho > f1
sort ./fr_eu/english_1_conxt > f2
python combine.py f1 f2  ./fr_eu/english_1_all
echo '12 done'
sort ./sp_eu/spanish_1_ortho > f1
sort ./sp_eu/spanish_1_conxt > f2
python combine.py f1 f2  ./sp_eu/spanish_1_all
echo '13 done'
sort ./sp_eu/spanish_2_ortho > f1
sort ./sp_eu/spanish_2_conxt > f2
python combine.py f1 f2  ./sp_eu/spanish_2_all
echo '14 done'
sort ./sp_eu/english_1_ortho > f1
sort ./sp_eu/english_1_conxt > f2
python combine.py f1 f2  ./sp_eu/english_1_all
echo '15 done'
sort ./sp_g/spanish_g_ortho > f1
sort ./sp_g/spanish_g_conxt > f2
python combine.py f1 f2 ./sp_g/spanish_g_all
