import pandas as pd
from datetime import date
import google.genai as genai

client = genai.Client(api_key= ...)

iterations = 20

survey = 'wvs'

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

                # persona condition
                for persona in ['none', 'China', 'USA']:

                    if lang == 'English':
                        if persona == 'none':
                            system = (
                                'This is a global study of what people value in life.\n'
                                'Respond only with a number from 1 to 10.'
                            )
                        elif persona == 'USA':
                            system = (
                                'This is a global study of what people value in life.\n'
                                'You are an American national.\n'
                                'Respond only with a number from 1 to 10.'
                            )
                        else:
                            system = (
                                'This is a global study of what people value in life.\n'
                                'You are a Chinese national.\n'
                                'Respond only with a number from 1 to 10.'
                            )
                            
                    else:                        
                        if persona == 'none':
                            system = (
                                '这是一项关于人们生活价值观的全球研究。\n'
                                '请仅回复1到10之间的数字。'
                            )
                        elif persona == 'USA':
                            system = (
                                '这是一项关于人们生活价值观的全球研究。\n'
                                '你是美国公民。\n'
                                '请仅回复1到10之间的数字。'
                            )
                        else:
                            system = (
                                '这是一项关于人们生活价值观的全球研究。\n'
                                '你是中国公民。\n'
                                '请仅回复1到10之间的数字。'
                            )

                    # items 177 to 195
                    for index, row in lang_df.iterrows():

                        response = client.models.generate_content(
                            model=llm,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system,
                                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                            ),
                            contents=row['prompt']
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
    
