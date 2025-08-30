import pandas as pd
from datetime import date
import google.genai as genai

client = genai.Client(api_key= ...)

iterations = 20

survey = 'mfq'

model_nationality = 'USA'

api = 'gemini'

temp = 1 # default

# load data
df = pd.read_csv(f'{survey}_items.csv')

try:

    # models
    for llm in ['gemini-2.5-flash', 'gemini-2.0-flash']:

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
                        
                        response = client.models.generate_content(
                            model=llm,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system,
                                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                            ),
                            contents=prompt + row['prompt']
                        )
        
                        results.append({
                            'survey': survey,
                            'LLM': llm,
                            'language': lang,
                            'persona': persona,
                            'item': row['item'],
                            'iteration': iteration,
                            'response': response.candidates[0].content.parts[0].text,
                            'model_fullname': response.model_version,
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
    