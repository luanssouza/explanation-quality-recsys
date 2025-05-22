python dataset_mapper_splitted.py --dataset ml1m
cd models/PGPR
python preprocess.py --dataset ml1m
python train_transe_model.py --dataset ml1m
python train_agent.py --dataset ml1m
python test_agent.py --dataset ml1m
cd ../../
mkdir ./logs
python main.py --dataset=ml1m --opt=softETD > logs/ml1m_softETD.log
python main.py --dataset=ml1m --opt=softSEP > logs/ml1m_softSEP.log
python main.py --dataset=ml1m --opt=softLIR > logs/ml1m_softLIR.log
python main.py --dataset=ml1m --opt=ETDopt > logs/ml1m_ETDopt.log
python main.py --dataset=ml1m --opt=SEPopt > logs/ml1m_SEPopt.log
python main.py --dataset=ml1m --opt=LIRopt > logs/ml1m_LIRopt.log
python main.py --dataset=ml1m --opt=ETD_SEP_opt > logs/ml1m_ETD_SEP_opt.log
python main.py --dataset=ml1m --opt=ETD_LIR_opt > logs/ml1m_ETD_LIR_opt.log
python main.py --dataset=ml1m --opt=SEP_LIR_opt > logs/ml1m_SEP_LIR_opt.log
python main.py --dataset=ml1m --opt=ETD_SEP_LIR_opt > logs/ml1m_ETD_SEP_LIR_opt.log