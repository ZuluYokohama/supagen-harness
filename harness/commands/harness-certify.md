---
name: harness-certify
description: External OPEN|STOP gate on a claim bundle JSON
---

# /harness-certify

```bash
python harness/certify/v1/certify.py path/to/bundle.json
python harness/certify/v1/certify.py --demo filmore
python harness/certify/v1/certify.py path/to/bundle.json -o cert.json
```

Only this gate may OPEN. Scouts/LLMs explore only.
