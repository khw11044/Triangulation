from scipy import io



# 데이터 파일 불러오기

root = 'S1/Seq1/'
mat_file = io.loadmat(root + 'annot.mat')

# print(mat_file)
# print(type(mat_file))

for raw in mat_file:
    print(raw)

# # 특정 변수 읽기
annot2 = mat_file['annot2']
annot3 = mat_file['annot3']
cameras = mat_file['cameras']
frames = mat_file['frames']
univ_annot3 = mat_file['univ_annot3']

print()