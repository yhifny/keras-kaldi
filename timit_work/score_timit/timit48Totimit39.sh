# -------------------------------------------------------
# -------------------------------------------------------
# 
# 
#	Syntax: MEGenLM.sh timit48_file timit39_file phone39.lst
#
# -------------------------------------------------------
# -------------------------------------------------------
# (c) Yasser H. Abdel-Haleem 2004


#	if(strLabel=="vcl")	return "sil";
#	if(strLabel=="cl")	return "sil";
#	if(strLabel=="pau")	return	"sil";
#	if(strLabel=="epi")	return	"sil";
#	if(strLabel=="el")	return	"l";
#	if(strLabel=="en")	return	"n";
#	if(strLabel=="zh")	return	"sh";
#	if(strLabel=="ao")	return	"aa";
#	if(strLabel=="ix")	return	"ih";
#	if(strLabel=="ax")	return	"ah";

#	cat $1    | sed -e 's/vcl/sil/g' > 1.txt
#	cat 1.txt | sed -e 's/cl/sil/g'	 > 2.txt
#	cat 2.txt | sed -e 's/pau/sil/g' > 3.txt
#	cat 3.txt | sed -e 's/epi/sil/g' > 4.txt
#	cat 4.txt | sed -e 's/el/l/g'	 > 5.txt
#	cat 5.txt | sed -e 's/en/n/g'	 > 6.txt
#	cat 6.txt | sed -e 's/zh/sh/g'	 > 7.txt
#	cat 7.txt | sed -e 's/ao/aa/g'   > 8.txt
#	cat 8.txt | sed -e 's/ix/ih/g'   > 9.txt
#	cat 9.txt | sed -e 's/ax/ah/g'   > $2

cat $1    | sed -e 's/vcl/sil/g' | sed -e 's/cl/sil/g' | sed -e 's/pau/sil/g' | sed -e 's/epi/sil/g' | \
			sed -e 's/el/l/g'    | sed -e 's/en/n/g'   | sed -e 's/zh/sh/g'   | sed -e 's/ao/aa/g'   | \
			sed -e 's/ix/ih/g'	 | sed -e 's/ax/ah/g'   > $2




cat $2 |  sed -e 's/ /\n/g'|sort -u > 1.txt
cat $3 |  sed -e 's/ /\n/g'|sort -u > 2.txt
diff 1.txt 2.txt

rm -f 1.txt 2.txt 
