#python make_user_feature.py /data/zhilian_job/table1_user.txt feature/user_feature
#python make_jd_feature.py /data/zhilian_job/table2_jd.txt feature/jd_feature 
#python get_user_job_profile.py feature/user_feature feature/jd_feature
#python join_feature.py feature/user_feature feature/user_profile feature/jd_feature normal
python compress_feature.py
python train.py
python test.py
