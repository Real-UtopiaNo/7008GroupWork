import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
url_D = 'https://scribehow.com/library/digitalization-tools'
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}
response_D = rq.get(url=url_D, headers=headers)
soup_D = BeautifulSoup(response_D.text, 'html.parser')
sections_D = ['1. Slack', '2. Microsoft Teams', '3. Zoom', '4. Flock', '5. Asana', '6. Hive ', '7. Trello', '8. Wrike', '9. Scribe', '10. Lucidchart', '11. Whatfix', '12. Nuclino', '13. Zendesk', '14. AskNicely', '15. HubSpot Service Hub', '16. Gladly', '17. Guru', '18. Document360', '19. Bloomfire', '20. Happeo', '21. Scoro', '22. Pipedrive', '23. Keap', '24. Freshdesk', '25. Softr ']
toolName = [section.split('. ', 1)[1] for section in sections_D]
data_D = []
index = 0
for section in sections_D:
    section_header = soup_D.find('h3', string=section)
    if section_header:
        next_sibling = section_header
        for i in range(5):
            next_sibling = next_sibling.find_next_sibling()
        if next_sibling and next_sibling.name == 'p':
            for a in next_sibling.find_all('a'):
                toolDetail = a.find_next_sibling(string=True).strip()
                data_D.append({'Question': 'What is '+toolName[index]+'?', 'Answer': toolName[index]+' '+toolDetail})
    index += 1
df_D = pd.DataFrame(data_D)
df_D['Answer'] = df_D['Answer'].str.replace('Â', '')
df_D.to_csv('digitalization_tools_detail.csv', index=False, encoding='utf-8-sig')

url_P = 'https://www.kato.app/articles/best-proptech-tools-for-commercial-property-landlords'
response_P = rq.get(url=url_P, headers=headers)
soup_P = BeautifulSoup(response_P.text, 'html.parser')
sections_P = ['Trustek', 'Bright Spaces', 'Coyote', 'Cherre', 'Proda', 'Architrave', 'Stak', 'Re-Leased', 'NavigatorCRE', 'WiredScore', 'FundRE', 'Least']
data_P = []
for section in sections_P:
    check = 0
    detail_P = ''
    section_header = soup_P.find('h6', string=section)
    if section_header:
        next_sibling = section_header
        next_sibling = next_sibling.find_next_sibling()
        if next_sibling and next_sibling.name == 'p':
            for a in next_sibling.find_all('a'):
                if check == 0:
                    if a.previous_sibling:
                        toolDetail_1 = a.previous_sibling.strip()
                    else:
                        toolDetail_1 = ''
                    toolDetail_2 = a.next_sibling.strip()
                    detail_P += (toolDetail_1 + ' ' + a.get_text() + ' ' + toolDetail_2 + ' ')
                else:
                    toolDetail_2 = a.next_sibling.strip()
                    detail_P += (a.get_text() + ' ' + toolDetail_2)
                check += 1
            data_P.append({'Question': 'The detail about '+section+'?', 'Answer': detail_P})
df_P = pd.DataFrame(data_P)
df_P['Answer'] = df_P['Answer'].str.replace('Â', '')
df_P['Answer'] = df_P['Answer'].str.replace('â', '')
df_P.to_csv('PropTech_tools_detail.csv', index=False, encoding='utf-8-sig')