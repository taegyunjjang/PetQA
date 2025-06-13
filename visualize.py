import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 제공해주신 텍스트 데이터를 구조화된 딕셔너리로 수동 파싱합니다.
# 이 딕셔너리는 DataFrame 생성에 사용됩니다.
data_raw = {
    'Expert': {
        'gpt-4o-mini': {
            'Dog': {
                'preprocessed': {'ROUGE': 0.261, 'BERTScore': 0.723, 'Factuality': 0.646},
                'raw': {'ROUGE': 0.265, 'BERTScore': 0.722, 'Factuality': 0.647}
            },
            'Cat': {
                'preprocessed': {'ROUGE': 0.271, 'BERTScore': 0.727, 'Factuality': 0.613},
                'raw': {'ROUGE': 0.270, 'BERTScore': 0.724, 'Factuality': 0.606}
            }
        },
        'claude-3-haiku': {
            'Dog': {
                'preprocessed': {'ROUGE': 0.248, 'BERTScore': 0.705, 'Factuality': 0.638},
                'raw': {'ROUGE': 0.245, 'BERTScore': 0.699, 'Factuality': 0.622}
            },
            'Cat': {
                'preprocessed': {'ROUGE': 0.251, 'BERTScore': 0.706, 'Factuality': 0.600},
                'raw': {'ROUGE': 0.250, 'BERTScore': 0.702, 'Factuality': 0.577}
            }
        },
        'gemini-2.0-flash': {
            'Dog': {
                'preprocessed': {'ROUGE': 0.253, 'BERTScore': 0.708, 'Factuality': 0.640},
                'raw': {'ROUGE': 0.248, 'BERTScore': 0.701, 'Factuality': 0.633}
            },
            'Cat': {'preprocessed': {'ROUGE': 0.252, 'BERTScore': 0.708, 'Factuality': 0.617},
                    'raw': {'ROUGE': 0.249, 'BERTScore': 0.701, 'Factuality': 0.604}}
        },
        'gemma-3-4b': {
            'Dog': {'preprocessed': {'ROUGE': 0.234, 'BERTScore': 0.694, 'Factuality': 0.602},
                    'raw': {'ROUGE': 0.232, 'BERTScore': 0.697, 'Factuality': 0.591}},
            'Cat': {'preprocessed': {'ROUGE': 0.238, 'BERTScore': 0.697, 'Factuality': 0.573},
                    'raw': {'ROUGE': 0.235, 'BERTScore': 0.699, 'Factuality': 0.564}}
        },
        'qwen-2.5-7b': {
            'Dog': {'preprocessed': {'ROUGE': 0.202, 'BERTScore': 0.707, 'Factuality': 0.540},
                    'raw': {'ROUGE': 0.206, 'BERTScore': 0.703, 'Factuality': 0.503}},
            'Cat': {'preprocessed': {'ROUGE': 0.211, 'BERTScore': 0.709, 'Factuality': 0.506},
                    'raw': {'ROUGE': 0.209, 'BERTScore': 0.706, 'Factuality': 0.516}}
        },
        'exaone-3.5-7.8b': {
            'Dog': {'preprocessed': {'ROUGE': 0.243, 'BERTScore': 0.715, 'Factuality': 0.634},
                    'raw': {'ROUGE': 0.243, 'BERTScore': 0.711, 'Factuality': 0.626}},
            'Cat': {'preprocessed': {'ROUGE': 0.246, 'BERTScore': 0.713, 'Factuality': 0.575},
                    'raw': {'ROUGE': 0.242, 'BERTScore': 0.708, 'Factuality': 0.575}}
        }
    },
    'Non Expert': {
        'gpt-4o-mini': {
            'Dog': {'preprocessed': {'ROUGE': 0.239, 'BERTScore': 0.713, 'Factuality': 0.487},
                    'raw': {'ROUGE': 0.245, 'BERTScore': 0.716, 'Factuality': 0.515}},
            'Cat': {'preprocessed': {'ROUGE': 0.239, 'BERTScore': 0.717, 'Factuality': 0.460},
                    'raw': {'ROUGE': 0.236, 'BERTScore': 0.716, 'Factuality': 0.467}}
        },
        'claude-3-haiku': {
            'Dog': {'preprocessed': {'ROUGE': 0.217, 'BERTScore': 0.693, 'Factuality': 0.478},
                    'raw': {'ROUGE': 0.216, 'BERTScore': 0.689, 'Factuality': 0.482}},
            'Cat': {'preprocessed': {'ROUGE': 0.221, 'BERTScore': 0.695, 'Factuality': 0.455},
                    'raw': {'ROUGE': 0.217, 'BERTScore': 0.691, 'Factuality': 0.433}}
        },
        'gemini-2.0-flash': {
            'Dog': {'preprocessed': {'ROUGE': 0.209, 'BERTScore': 0.690, 'Factuality': 0.457},
                    'raw': {'ROUGE': 0.209, 'BERTScore': 0.685, 'Factuality': 0.478}},
            'Cat': {'preprocessed': {'ROUGE': 0.219, 'BERTScore': 0.697, 'Factuality': 0.446},
                    'raw': {'ROUGE': 0.213, 'BERTScore': 0.688, 'Factuality': 0.440}}
        },
        'gemma-3-4b': {
            'Dog': {'preprocessed': {'ROUGE': 0.194, 'BERTScore': 0.679, 'Factuality': 0.430},
                    'raw': {'ROUGE': 0.197, 'BERTScore': 0.681, 'Factuality': 0.405}},
            'Cat': {'preprocessed': {'ROUGE': 0.199, 'BERTScore': 0.687, 'Factuality': 0.414},
                    'raw': {'ROUGE': 0.196, 'BERTScore': 0.687, 'Factuality': 0.414}}
        },
        'qwen-2.5-7b': {
            'Dog': {'preprocessed': {'ROUGE': 0.197, 'BERTScore': 0.709, 'Factuality': 0.421},
                    'raw': {'ROUGE': 0.196, 'BERTScore': 0.705, 'Factuality': 0.415}},
            'Cat': {'preprocessed': {'ROUGE': 0.207, 'BERTScore': 0.714, 'Factuality': 0.383},
                    'raw': {'ROUGE': 0.206, 'BERTScore': 0.709, 'Factuality': 0.386}}
        },
        'exaone-3.5-7.8b': {
            'Dog': {'preprocessed': {'ROUGE': 0.211, 'BERTScore': 0.700, 'Factuality': 0.435},
                    'raw': {'ROUGE': 0.206, 'BERTScore': 0.697, 'Factuality': 0.442}},
            'Cat': {'preprocessed': {'ROUGE': 0.210, 'BERTScore': 0.704, 'Factuality': 0.416},
                    'raw': {'ROUGE': 0.203, 'BERTScore': 0.697, 'Factuality': 0.394}}
        }
    }
}

# 중첩된 딕셔너리를 pandas DataFrame으로 평탄화합니다.
flat_data = []
for expertise, models_data in data_raw.items():
    for model, animals_data in models_data.items():
        for animal, process_types_data in animals_data.items():
            for process_type, metrics_data in process_types_data.items():
                row = {
                    'Expertise': expertise,
                    'Model': model,
                    'Animal': animal,
                    'ProcessType': process_type,
                    **metrics_data # ROUGE, BERTScore, Factuality 지표들을 언팩하여 추가
                }
                flat_data.append(row)

df_all = pd.DataFrame(flat_data)

# 성능 시각화를 위한 함수를 정의합니다.
def plot_performance(df_data, title, file_prefix):
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.5

    # 스택 바 차트를 위한 지표별 시작 위치 계산
    df_data['BERTScore_bottom'] = 0
    df_data['Factuality_bottom'] = df_data['BERTScore']
    df_data['ROUGE_bottom'] = df_data['BERTScore'] + df_data['Factuality']

    # zorder를 설정하여 막대가 그리드 선 위에 오도록 플로팅합니다.
    ax.bar(df_data['Model'], df_data['BERTScore'], bar_width, label='BERTScore', color='#6cace4', bottom=df_data['BERTScore_bottom'], zorder=3)
    ax.bar(df_data['Model'], df_data['Factuality'], bar_width, label='Factuality', color='#e77471', bottom=df_data['Factuality_bottom'], zorder=3)
    ax.bar(df_data['Model'], df_data['ROUGE'], bar_width, label='ROUGE', color='#fdd069', bottom=df_data['ROUGE_bottom'], zorder=3)

    # y축 눈금 및 그리드 선을 설정합니다.
    y_ticks = np.arange(0, 2.25, 0.25) # 0부터 2.0까지 0.25 간격으로 눈금 설정
    ax.set_yticks(y_ticks)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0) # 그리드 선은 실선으로, 투명도 및 zorder 설정

    # 레이블 및 제목 설정
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 2.0) # y축 범위 설정

    # x축 레이블 회전
    plt.xticks(rotation=45, ha='right')

    # 범례 표시
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout() # 레이아웃 자동 조정
    plt.savefig(f"visualization_results/{file_prefix}.png")
    plt.close(fig) # 메모리 해제를 위해 figure 닫기

# 생성할 4가지 시각화의 설정을 정의합니다.
plot_configs = [
    {'expertise': 'Expert', 'animal': 'Dog', 'process_type': 'preprocessed', 'title': 'Expert - Dog - Preprocessed'},
    {'expertise': 'Expert', 'animal': 'Cat', 'process_type': 'preprocessed', 'title': 'Expert - Cat - Preprocessed'},
    {'expertise': 'Non Expert', 'animal': 'Dog', 'process_type': 'preprocessed', 'title': 'Non Expert - Dog - Preprocessed'},
    {'expertise': 'Non Expert', 'animal': 'Cat', 'process_type': 'preprocessed', 'title': 'Non Expert - Cat - Preprocessed'},
    {'expertise': 'Expert', 'animal': 'Dog', 'process_type': 'raw', 'title': 'Expert - Dog - Raw'},
    {'expertise': 'Expert', 'animal': 'Cat', 'process_type': 'raw', 'title': 'Expert - Cat - Raw'},
    {'expertise': 'Non Expert', 'animal': 'Dog', 'process_type': 'raw', 'title': 'Non Expert - Dog - Raw'},
    {'expertise': 'Non Expert', 'animal': 'Cat', 'process_type': 'raw', 'title': 'Non Expert - Cat - Raw'}
]

# 각 설정에 따라 데이터를 필터링하고 플로팅 함수를 호출합니다.
for i, config in enumerate(plot_configs):
    title = config['title']
    subset_df = df_all[
        (df_all['Expertise'] == config['expertise']) &
        (df_all['Animal'] == config['animal']) &
        (df_all['ProcessType'] == config['process_type'])
    ].copy() # SettingWithCopyWarning을 피하기 위해 .copy() 사용

    plot_performance(subset_df, config['title'], f"{title}")