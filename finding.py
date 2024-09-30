import pandas as pd
import subprocess

'''columns=['Id', 'Recomendations']

data = [
['Tiger Lily', 'Before putting a flower arrangement in a container with water, it must be watered. '
                     'After that, watering should be carried out every few days. It is not necessary to spray the bouquet, so '
                     'both plants get enough moisture anyway. From time to time, you need to remove it from the composition '
                     'wilted little daisies, as they will only take away nutrients from other parts of the plant. '
                     'We should not forget about the replacement of water. You need to update it 1 time every 1-2 days. '
                     'At the same time, it is better to use settled water at room temperature.'],
['Mexican aster', 'Funny bouquet))'],
['Rose', 'Put a bouquet of roses in a place with a good supply of fresh air, but without drafts, away from heaters,'
          'air conditioners and direct sunlight. Spray the leaves daily with water from a spray bottle, avoiding contact '
          'on the buds. Change the water every day and prune the flowers.'],
['Sunflower', 'Sunflowers are thermophilic — do not put them in cold water. '
                'Add a pinch of salt to the water to keep the petals elastic. '
                'Change the water in the vase every 2-3 days, rinsing the vase well with detergent.'
                'Be sure to make an oblique incision on the stem with an inclination of 45°.'],
['King Protea', 'Even when it is cold, cut tulips are fragile flowers. Sunlight is the main enemy of tulips: '
           'the flower will wither, fade, and the petals will gradually fall off. The place for a bouquet of tulips should be cool,'
           'well ventilated, not exposed to direct sunlight. Do not place a vase of flowers next to heaters '
           'or hot air currents.']
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(r'recomendations.csv', encoding='cp1251')'''



def find_info(src):
    df = pd.read_csv('recomendations.csv', encoding='cp1251')

    predict_val = subprocess.getoutput('python recognition.py --filepath ' + src + ' --checkpoint safi.h5').split('\n')

    kol = df[df['Id'] == predict_val[len(predict_val) - 1]].index[0]

    val = df.loc[kol, 'Recomendations']

    return predict_val[len(predict_val) - 1], val



