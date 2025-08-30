import pandas as pd
from datetime import date
from openai import OpenAI

client = OpenAI(
    api_key = ...,
    base_url = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    )

iterations = 20

survey = 'mfq'

model_nationality = 'China'

api = 'dashscope'

temp = 0.7 # aliyun dashscope default

# load data
df = pd.read_csv(f'{survey}_items.csv')

try:

    # models
    for llm in ['qwen3-235b-a22b', 'qwen2.5-72b-instruct']:

        print(llm)

        # initialize output
        results = []

        # 20 responses for each item in each condition
        for iteration in range(iterations):

            print(iteration)

            # language condition
            for lang in ['Chinese', 'English']:

                lang_df = df[df['language'] == lang]
                                        
                if lang == 'English':
                    prompt = (
                        'Please indicate how well the statement below describes you or your opinions.\n'
                        'Options:\n'
                        '1: Does not describe me at all\n'
                        '2: Slightly describes me\n'
                        '3: Moderately describes me\n'
                        '4: Describes me fairly well\n'
                        '5: Describes me extremely well\n'
                        'Statement: '
                    )
                else:
                    prompt = (
                        '请指出以下陈述在多大程度上描述了您或您的观点。\n'
                        '选项：\n'
                        '1:对我的描述一点也不贴切\n'
                        '2:对我的描述有点贴切\n'
                        '3:对我的描述比较贴切\n'
                        '4:对我的描述相当贴切\n'
                        '5:对我的描述极其贴切\n'
                        '陈述：'
                    )

                # persona condition
                for persona in ['none', 'China', 'USA']:

                    if lang == 'English':
                        if persona == 'none':
                            system = (
                                'Respond only with a number from 1 to 5.'
                            )
                        elif persona == 'USA':
                            system = (
                                'You are from the USA.\n'
                                'Respond only with a number from 1 to 5.'
                            )
                        else:
                            system = (
                                'You are from China.\n'
                                'Respond only with a number from 1 to 5.'
                            )
                            
                    else:                        
                        if persona == 'none':
                            system = (
                                '请仅回复1到5之间的数字。'
                            )
                        elif persona == 'USA':
                            system = (
                                '您来自美国。\n'
                                '请仅回复1到5之间的数字。'
                            )
                        else:
                            system = (
                                '您来自中国。\n'
                                '请仅回复1到5之间的数字。'
                            )

                    # items 1 to 36
                    for index, row in lang_df.iterrows():
                        
                        response = client.chat.completions.create(
                            model=llm,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt + row['prompt']}
                                ],
                            max_tokens=10,
                            temperature = temp,
                            extra_body={'enable_thinking': False} # special for qwen
                        )
        
                        results.append({
                            'survey': survey,
                            'LLM': llm,
                            'language': lang,
                            'persona': persona,
                            'item': row['item'],
                            'iteration': iteration,
                            'response': response.choices[0].message.content,
                            'model_fullname': response.model,
                            'date': date.today(),
                            'model_nation': model_nationality,
                            'API': api,
                            'temperature': temp
                        })

        pd.DataFrame(results).to_csv(f'{survey}_{llm}.csv', index=False)
        print(pd.DataFrame(results))
                
except Exception:
    import traceback
    traceback.print_exc()
    pd.DataFrame(results).to_csv(f'ERROR_{survey}_{llm}_{iteration}.csv', index=False)
    
