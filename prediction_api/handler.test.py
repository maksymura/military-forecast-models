from datetime import datetime

import handler

res = handler.predict("Kyiv", datetime(2022, 7, 3))

print(res)
