import glob
import os
import pprint

nums = set()
for fn in glob.glob('../data/checkpoints/-*.*'):
    fn = fn[fn.find('-'):]
    try:
        num = int(fn[1:fn.find('.')])
        if num % 1000000 != 0:
            for delete_fn in glob.glob('../data/checkpoints/-'+str(num)+'.*'):
                os.remove(delete_fn)
        else:
            nums.add(num)
    except:
        pass

pprint.PrettyPrinter().pprint(sorted(nums))