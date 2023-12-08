import openai
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
import time
import datetime


class Identifier(ABC):
    """ Abstract Moderator Class. """
    def __init__(self):
        return

    def annotate_protected_attributes(self, **kwargs):
        return


class ChatGPTIdentifier(Identifier):
    def __init__(self, openai_api_key="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"):
        super().__init__()
        openai.api_key = openai_api_key
        return

    def annotate_protected_attributes(self, tokens, temperature=0.3):
        """  ."""
        result = []
        raw_responses = []
        exception_logs = []

        for tk in tqdm(tokens):
            chatgpt_response = ""
            try:
                chatgpt_response = self._chatgpt_annotate(tk)
                #print(chatgpt_response)
                #print("\n")
                raw_responses.append(f"{tk}: {chatgpt_response}\n")
                chatgpt_main_category, chatgpt_category_main_score, chatgpt_explanation = chatgpt_response.split('|')

                result.append({'word': tk,
                               'chatgpt_protected_category': chatgpt_main_category,  # Protected category annotated by chatgpt
                               'chatgpt_protected_category_score': chatgpt_category_main_score,  # Reliability score of the protected category
                               'chatgpt_explanation': chatgpt_explanation  # Explanation provided for tha annotated category
                               })
            except Exception as ex:
                try:
                    chatgpt_response = chatgpt_response.split('\n')[2]
                    chatgpt_main_category, chatgpt_category_main_score, chatgpt_explanation = chatgpt_response.split('|')
                    result.append({'word': tk,
                                   'chatgpt_protected_category': chatgpt_main_category,  # Protected category annotated by chatgpt
                                   'chatgpt_protected_category_score': chatgpt_category_main_score,  # Reliability score of the protected category
                                   'chatgpt_explanation': chatgpt_explanation  # Explanation provided for tha annotated category
                                   })
                except Exception as ex:
                    print(f"{ex} in {tk}!")
                    #print({chatgpt_response})
                    #print("\n")
                    exception_logs.append(f"Exception in: {tk}: {chatgpt_response}\n")

        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
        filename = f"ChatGPT-annotation-{timestamp}"

        df_annotated = pd.DataFrame(result)

        df_cleaned = self._clean_chatgpt_responses(df_annotated)

        result, acc_list = self._aggregate_protected_categories_votes(df_cleaned)

        print(result)

        print(acc_list)

        print()

        #df_annotated.to_csv(f"chatgpt_annotation/{filename}.csv", index=None)

        #with open(f'chatgpt_annotation/raw_responses/{filename}.txt', 'w') as responses_file:
        #    responses_file.writelines(raw_responses)

        #with open(f'chatgpt_annotation/log/{filename}.txt', 'w') as responses_file:
        #    responses_file.writelines(exception_logs)

        return df_annotated

    def _clean_chatgpt_responses(self, df_raw,  protected_category_column_name="chatgpt_protected_category"):
        """ Cleans the chatgpt response.
        Args:
            df (pd.DataFrame): The dataframe containing the chatgpt response.
        """
        df = df_raw.copy()
        df["word"] = df["word"].str.lower()
        df["word"] = df["word"].str.strip()
        df[protected_category_column_name] = df[protected_category_column_name].str.lower()
        df[protected_category_column_name] = df[protected_category_column_name].str.strip()

        # Replace the "None" category with "none-category"
        df[protected_category_column_name].replace('none', 'none-category', inplace=True)
        df[protected_category_column_name] = df[protected_category_column_name].fillna("none-category")

        # Correct errors in the response provided by chatgpt for some categories
        df[protected_category_column_name].replace('"pregnancy and maternity', 'pregnancy and maternity', inplace=True)
        df[protected_category_column_name].replace('"disability', 'disability', inplace=True)
        df[protected_category_column_name].replace('"race', 'race', inplace=True)
        #df[protected_category_column_name].replace('proper name', 'none-category', inplace=True)
        return df

    @staticmethod
    def _aggregate_protected_categories_votes(df, protected_category_column_name="chatgpt_protected_category"):
        # Compute the counts in a single grouped operation
        grouped = df.groupby('word')[protected_category_column_name].value_counts().unstack(fill_value=0)

        # Add a total count column
        grouped['count_tot'] = grouped.sum(axis=1)

        # Check if 'none-category' column exists and rename it to 'count_non_pa'
        if 'none-category' in grouped.columns:
            grouped = grouped.rename(columns={'none-category': 'count_non_pa'})
            # Calculate 'count_pa' (which is count_tot - count_non_pa)
            grouped['count_pa'] = grouped['count_tot'] - grouped['count_non_pa']
        else:
            # If 'none-category' column doesn't exist, set 'count_non_pa' to 0 and 'count_pa' to 'count_tot'
            grouped['count_non_pa'] = 0
            grouped['count_pa'] = grouped['count_tot']

        # Reset the index to turn 'word' back into a column
        grouped = grouped.reset_index()

        # Select only the required columns
        result = grouped[['word', 'count_pa', 'count_non_pa', 'count_tot']]

        # Convert to a list of dictionaries, if needed
        acc_list = result.to_dict('records')
        return result, acc_list

    @staticmethod
    def _chatgpt_annotate(tk, temperature=0.3, chatgpt_model="gpt-3.5-turbo"):
        """ Performs annotation using ChatGPT of the input token.
        Args:
            tk (str): The token to be annotated.
            temperature (float): The temperature of the ChatGPT model.
            chatgpt_model (str): The ChatGPT model to be used.
        """
        completion = openai.chat.completions.create(
            model=chatgpt_model,
            messages=[
                {'role': 'user', 'content': '''Consider these 9 protected categories defined by the Equality Act law to avoid discrimination of automatic decision-making algorithms:
                "Age": A person belonging to a particular age (for example, 32 year olds) or range of ages (for example 18 to 30 year olds).
                "Disability": A person has a disability if she or he has a physical or mental impairment which has a substantial and long-term adverse effect on that person's ability to carry out normal day-to-day activities.
                "Gender reassignment": The process of transitioning from one sex to another.
                "Marriage and civil partnership": Marriage is a union between a man and a woman or between a same-sex couple. Same-sex couples can also have their relationships legally recognised as 'civil partnerships'. Civil partners must not be treated less favourably than married couples.
                "Pregnancy and maternity": Pregnancy is the condition of being pregnant or expecting a baby. Maternity refers to the period after the birth, and is linked to maternity leave in the employment context. In the non-work context, protection against maternity discrimination is for 26 weeks after giving birth, and this includes treating a woman unfavourably because she is breastfeeding.
                "Race": Refers to the protected characteristic of race. It refers to a group of people defined by their race, colour, and nationality (including citizenship) ethnic or national origins.
                "Religion and belief": Religion refers to any religion, including a lack of religion. Belief refers to any religious or philosophical belief and includes a lack of belief. Generally, a belief should affect your life choices or the way you live for it to be included in the definition.
                "Sex": A word that explicitly refers to the gender of a person: e.g., man or woman, she or he, mr or mrs, male of female, madame etc.
                "Sexual orientation": Whether a person's sexual attraction is towards their own sex, the opposite sex or to both sexes.
                '''},
                {'role': 'user', 'content': """You can learn more about the discriminations along each protected characteristic on the following URLs:
                "Age" : https://www.equalityhumanrights.com/en/advice-and-guidance/age-discrimination
                "Disability":https://www.equalityhumanrights.com/en/disability-advice-and-guidance
                "Gender reassignment": https://www.equalityhumanrights.com/en/advice-and-guidance/gender-reassignment-discrimination
                "Marriage and civil partnership": https://www.equalityhumanrights.com/en/advice-and-guidance/marriage-and-civil-partnership-discrimination
                "Pregnancy and maternity": https://www.equalityhumanrights.com/en/node/5916
                "Race": https://www.equalityhumanrights.com/en/advice-and-guidance/race-discrimination
                "Religion and belief": https://www.equalityhumanrights.com/en/religion-or-belief-work
                "Sex": https://www.equalityhumanrights.com/en/advice-and-guidance/sex-discrimination
                "Sexual orientation": https://www.equalityhumanrights.com/en/advice-and-guidance/sexual-orientation-discrimination   """},

                {'role': 'user', 'content': f"""Given the previously defined protected categories "Age", "Disability", "Gender reassignment", "Marriage and civil partnership", "Pregnancy and maternity", "Race", "Religion and belief", "Sex", "Sexual orientation". 
                How would you classify the word "{tk}" and which [0,100] reliability scores (only one) would you give to your assessment? You must assign one category. 
                If a word does not fit any categories, you must assign the category "None" with the reliability score and the relative explanation. 
                Provide the answer in the format: "Protected Category|Reliability Score from 0 to 100 for the protected category|Explanation of why the word belong to the protected category.". 
                In case of a word does not fall into any category, provide the answer in the format: "None|Reliability Score from 0 to 100 for the None category|Explanation of why the word does not fall under any of the defined protected categories. 
                Each answer MUST have exactly two | symbols and only one line; otherwise, I cannot process your response. 
                Please annotate a word to a protected category only if it is strictly related. """}
            ],
            temperature=temperature,
            max_tokens=150  # Adjust as needed
        )

        return completion.choices[0].message.content

    @staticmethod
    def _chatgpt_annotate_old(tk, temperature=0.3, chatgpt_model="gpt-3.5-turbo"):
        """ Performs annotation using ChatGPT of the input token.
        Args:
            tk (str): The token to be annotated.
            temperature (float): The temperature of the ChatGPT model.
            chatgpt_model (str): The ChatGPT model to be used.
        """
        response = openai.ChatCompletion.create(
            model=chatgpt_model,
            messages=[
                {'role': 'user', 'content': '''Consider these 9 protected categories defined by the Equality Act law to avoid discrimination of automatic decision-making algorithms:
                "Age": A person belonging to a particular age (for example, 32 year olds) or range of ages (for example 18 to 30 year olds).
                "Disability": A person has a disability if she or he has a physical or mental impairment which has a substantial and long-term adverse effect on that person's ability to carry out normal day-to-day activities.
                "Gender reassignment": The process of transitioning from one sex to another.
                "Marriage and civil partnership": Marriage is a union between a man and a woman or between a same-sex couple. Same-sex couples can also have their relationships legally recognised as 'civil partnerships'. Civil partners must not be treated less favourably than married couples.
                "Pregnancy and maternity": Pregnancy is the condition of being pregnant or expecting a baby. Maternity refers to the period after the birth, and is linked to maternity leave in the employment context. In the non-work context, protection against maternity discrimination is for 26 weeks after giving birth, and this includes treating a woman unfavourably because she is breastfeeding.
                "Race": Refers to the protected characteristic of race. It refers to a group of people defined by their race, colour, and nationality (including citizenship) ethnic or national origins.
                "Religion and belief": Religion refers to any religion, including a lack of religion. Belief refers to any religious or philosophical belief and includes a lack of belief. Generally, a belief should affect your life choices or the way you live for it to be included in the definition.
                "Sex": A word that explicitly refers to the gender of a person: e.g., man or woman, she or he, mr or mrs, male of female, madame etc.
                "Sexual orientation": Whether a person's sexual attraction is towards their own sex, the opposite sex or to both sexes.
                '''},
                {'role': 'user', 'content': """You can learn more about the discriminations along each protected characteristic on the following URLs:
                "Age" : https://www.equalityhumanrights.com/en/advice-and-guidance/age-discrimination
                "Disability":https://www.equalityhumanrights.com/en/disability-advice-and-guidance
                "Gender reassignment": https://www.equalityhumanrights.com/en/advice-and-guidance/gender-reassignment-discrimination
                "Marriage and civil partnership": https://www.equalityhumanrights.com/en/advice-and-guidance/marriage-and-civil-partnership-discrimination
                "Pregnancy and maternity": https://www.equalityhumanrights.com/en/node/5916
                "Race": https://www.equalityhumanrights.com/en/advice-and-guidance/race-discrimination
                "Religion and belief": https://www.equalityhumanrights.com/en/religion-or-belief-work
                "Sex": https://www.equalityhumanrights.com/en/advice-and-guidance/sex-discrimination
                "Sexual orientation": https://www.equalityhumanrights.com/en/advice-and-guidance/sexual-orientation-discrimination   """},

                {'role': 'user', 'content': f"""Given the previously defined protected categories "Age", "Disability", "Gender reassignment", "Marriage and civil partnership", "Pregnancy and maternity", "Race", "Religion and belief", "Sex", "Sexual orientation". 
                How would you classify the word "{tk}" and which [0,100] reliability scores (only one) would you give to your assessment? You must assign one category. 
                If a word does not fit any categories, you must assign the category "None" with the reliability score and the relative explanation. 
                Provide the answer in the format: "Protected Category|Reliability Score from 0 to 100 for the protected category|Explanation of why the word belong to the protected category.". 
                In case of a word does not fall into any category, provide the answer in the format: "None|Reliability Score from 0 to 100 for the None category|Explanation of why the word does not fall under any of the defined protected categories. 
                Each answer MUST have exactly two | symbols and only one line; otherwise, I cannot process your response. 
                Please annotate a word to a protected category only if it is strictly related. """}
            ],
            temperature=temperature,
        )

        return response.choices[0]['message']['content']
