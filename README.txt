1��make_lmdb.py
���ļ�Ϊ����lmdb���ݼ��Ĵ���
key���ļ�·����value���ļ��Ķ������������ݣ��ɲ���opencv�е�imdecode���н���
ע��lmdb��key��value��Ϊbytes��ʽ

��һ�ַ�ʽ����ͼƬ��numpy����תΪbytes��������shape��dtypeһ���洢��lmdb�������ַ�ʽû�в����κ�ѹ���㷨����ʹlmdb��úܴ���˲���ȡ��

2��utils/image_process.py
�����������һ������LaneDatasetLMDB
�����ڳ�ʼ���׶������lmdb�ĳ�ʼ�����룬��ȡ����ʱ��lmdb�ж�ȡ�ļ��������ֽ���������imdecode���н��룬��ע��RGBͼƬ�ͻҶ�ͼ�Ĳ�ͬ��

3��train.py
��LaneDataset�޸�ΪLaneDatasetLMDB���轫lmdb�ļ�·�����롣
ע����windows��mac�У����ö���̶�ȡlmdbʱ�������linux���������ȡ��