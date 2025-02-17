import json

layers: dict[str, dict] = {}

with open("test.tmp", "r", encoding="utf-8") as f:
    #
    txt: str = f.read()



data_layer: dict
txt_json: str
i: int = 0
j: int = 0
i = txt.find("```json\n", j)
while i != -1:
    #
    j = txt.find("```\n", i + 3)
    #
    if j == -1:
        break
    #
    txt_json = txt[i+8:j]
    #
    try:
        data_layer = json.loads(txt_json)
    except Exception as e:
        print(f"ERROR during json on \n\n{txt_json}\n\n{e}")
        break
    #
    layers[data_layer["layer_name"]] = data_layer
    #
    i = txt.find("```json\n", j)

#
with open("layers.json", "w", encoding="utf-8") as f:
    json.dump(layers, f)

