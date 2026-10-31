# Demo script — midterm presentation

Six real issues from the **held-out test split**. The model has never seen any of them,
and the team's actual estimate is known. Paste the title and description into the demo,
select the project, press Estimate.

Measured over a 400-issue sample of the test split:
**26% exact matches, 49% within one story point.** Know these numbers — "how often is it
right?" is a likely question.

---

## 1. duracloud — actual **1**, predicted **1**

**Project:** `duracloud`

**Title:**
```
Retrieval Tool: Hidden password option
```

**Description:**
```
DURACLOUD-759 added a hidden password for the Sync Tool. This task is to make use of the same environment variable to pick up the password for the Retrieval Tool. This will help to keep the functionality of the two tools in sync.
```

Similar issues returned: 1, 4 and 2 points. Good opener — small task, correctly called small.

---

## 2. bamboo — actual **2**, predicted **2**

**Project:** `bamboo`

**Title:**
```
Different table for Quarantined test results on Result Summary
```

**Description:** *(leave empty)*

Worth doing one with no description — it shows the model works from the title alone, and
`has_description` is a real feature in the model.

---

## 3. bamboo — actual **3**, predicted **3**

**Project:** `bamboo`

**Title:**
```
Styling for new combined Plan Status and History Navigator
```

**Description:** *(leave empty)*

Top similar issue is "styling for tabs for build configuration" at 5 points. Good moment
to point out the similarity search found a genuinely comparable task, not just shared
keywords.

---

## 4. appceleratorstudio — actual **5**, predicted **5**

**Project:** `appceleratorstudio`

**Title:**
```
3.3.0 Studio Installer: Studio is installed in the wrong location
```

**Description:**
```
The 3.3.0 Studio installer downloads the temporary files into the AppData folder. After the temp files have been obtained it moves the entire program into the Program Files folder. Once all the files have been moved it attempts to remove the temp files, but fails to remove the root folder.
```

Raw output 5.11 — almost exactly on the scale point. Nice one to show the raw value next
to the rounded one.

---

## 5. appceleratorstudio — actual **8**, predicted **8**

**Project:** `appceleratorstudio`

**Title:**
```
Update content assist to support setting ActionBar properties via the Menu tag
```

**Description:**
```
Alloy 1.4 adds support for setting Android ActionBar properties (title, subtitle, etc.) as attributes of the Menu tag. This change affects Android only.
```

Raw 6.95, rounded to 8. The third similar issue is also 8 points.

---

## 6. clover — actual **20**, predicted **3** — THE MISS

**Project:** `clover`

**Title:**
```
as a user I want to use Clover with SBT
```

**Description:**
```
Integration of Clover-for-Scala with SBT. Tasks: write build plugin for SBT - equivalent of clover2:setup, shall intercept any scalac calls; shall add clover.jar to class paths for compilation and runtime.
```

**Show this one.** It is a 20-point epic that reads like a small task, and the model calls
it 3. Volunteering a failure is far stronger than being caught by one.

What to say: the model estimates from how an issue is *written*, and an epic written in one
line looks small. The failure mode is systematic — short text describing large scope — and
it is the clearest argument for the tool being a suggestion a manager reviews rather than a
decision it makes. Note that the third similar issue returned is 40 points: the explanation
panel is showing the warning sign even where the number does not.

---

## Running it

```
cd /d D:\MIT\Project\pm-insight
.venv\Scripts\activate
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

**Start it before the presentation, not during.** First load takes about a second while
the 45 MB forest loads; after that each estimate is roughly 40 ms.

---

## Questions to expect

**"Can it give hours instead of story points?"**
No. The dataset records story points, and converting to hours needs a team-specific
velocity that isn't in the data.

**"Why do I have to pick a project?"**
The selected model uses a one-hot project feature, so it only knows these 16. A second
variant was also trained on text alone — slightly less accurate, but it works for a project
the model has never seen. A real deployment trade-off, not an oversight.

**"How often is it right?"**
26% exact, 49% within one point on the test sample. Overall MAE 2.92 against the median
baseline's 2.98.

**"Is this better than just using the median?"**
By about 2%. The criterion was met on 9 of 16 projects, but say the size of the gain
honestly — the model earns its place on high-variance projects, not everywhere.
