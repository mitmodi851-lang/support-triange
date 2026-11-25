"""Create the SIMULATED training tickets (no real customer data is used).

Each category has hand-written templates with a built-in urgency label.
Random greetings, endings and slot values (plan names, amounts ...) create
several variants of each template. Every row keeps its `template_id`, so we
can split by template later and avoid leakage between train and test.

Run:  python -m triage.data_generation
"""
import random
import pandas as pd

from triage.config import SEED, TRAIN_PATH

# Urgency guide used for ALL labels:
#   High   = service blocked, money at risk, security risk, or a clear deadline within ~1 day
#   Medium = something is wrong but there is a workaround or a few days of time
#   Low    = information request or minor/cosmetic matter
TEMPLATES = {
    "Billing & Payments": [
        ("Medium", "My invoice for {month} shows {amount} but I expected a lower price. Can you explain the charges?"),
        ("High", "I was charged twice for my {plan} plan today and the money has left my bank account. Please fix this immediately."),
        ("Low", "Could you send me a copy of last month's invoice for my accounts department?"),
        ("Low", "How do I update the credit card used for my {plan} subscription?"),
        ("Medium", "The tax line on my invoice looks wrong. We need a corrected GST invoice before the filing date on Friday."),
        ("Medium", "Payment failed at checkout even though my card works elsewhere. I need to complete this order today."),
        ("High", "I see an unknown charge of {amount} on my statement from your company. I think my card was used without permission."),
        ("Low", "Please change my billing cycle from monthly to yearly. No rush."),
        ("High", "My auto-renewal payment was declined and my account will be suspended tomorrow unless I pay."),
        ("Low", "Where can I find the bank details for paying by bank transfer?"),
    ],
    "Technical Issue": [
        ("Medium", "The app crashes every time I open the reports page. It happens on both my laptop and phone."),
        ("High", "Our whole team cannot use the dashboard right now. It shows a server error 500 and we have a client demo in an hour."),
        ("Low", "How can I export my data to a CSV file? I could not find the option."),
        ("Low", "The mobile app is slow when loading images but it still works."),
        ("Medium", "Notifications stopped arriving after the latest update. Please investigate when you can."),
        ("High", "The API returns a timeout error for all requests since this morning and our production service depends on it."),
        ("Medium", "The search feature returns no results for anything I type, although everything else works."),
        ("Low", "I would like to know whether the {product} software supports dark mode."),
        ("High", "File uploads fail with an unknown error and I cannot submit my assignment before the deadline tonight."),
        ("Low", "The text on the settings page is cut off on my small phone screen."),
    ],
    "Account Access": [
        ("Medium", "I forgot my password and the reset email is not coming to my inbox."),
        ("High", "My account has been locked after too many attempts and I cannot get in to finish urgent work."),
        ("Low", "I want to change the email address linked to my profile."),
        ("High", "Someone logged into my account from another country. I did not do this and I need it secured right now."),
        ("Medium", "The two-factor authentication code is not being accepted on my new phone."),
        ("Low", "How do I add a new team member to our workspace?"),
        ("Medium", "Please delete my account and all my personal data."),
        ("Low", "My username shows the wrong name. How can I correct it?"),
        ("Medium", "I get an invalid credentials message even though I am sure my password is correct. It has been two days."),
        ("High", "I cannot sign in with my company single sign-on and the whole department is blocked."),
    ],
    "Shipping & Delivery": [
        ("Medium", "My order {order_id} was supposed to arrive last Monday and tracking has not updated for five days."),
        ("Medium", "The parcel for order {order_id} shows delivered but I never received it."),
        ("Medium", "Can I change the delivery address for my order {order_id}? It has not shipped yet."),
        ("Low", "What are your delivery charges to {city}?"),
        ("Medium", "The package arrived with a broken box and the item inside is damaged."),
        ("High", "I need my order {order_id} by tomorrow for a medical appointment and it is still not dispatched."),
        ("Low", "Do you ship internationally and how long does it usually take?"),
        ("Medium", "The courier left my parcel with a neighbour I do not know. Please confirm where it is."),
        ("Medium", "I received the wrong item in my order {order_id}. I ordered a blue one and got a red one."),
        ("High", "Perishable goods in my order {order_id} are stuck at a depot and will spoil today."),
    ],
    "Refund & Cancellation": [
        ("Medium", "I want to cancel my {plan} subscription before it renews next week."),
        ("Medium", "Please refund my order {order_id}; the product does not match the description."),
        ("Low", "How long does a refund take to reach my bank account?"),
        ("Low", "What is your return policy for opened items?"),
        ("Medium", "I cancelled my order two weeks ago and still have not received my refund of {amount}."),
        ("High", "I was charged after I had already cancelled my plan. I want that money returned today or I will raise a dispute with my bank."),
        ("Medium", "Please stop my free trial from converting to a paid plan."),
        ("Low", "Can I exchange the shoes for a bigger size instead of returning them?"),
        ("High", "I returned the laptop three weeks ago and the refund of {amount} has still not arrived. It is a large amount and I need it urgently."),
        ("Low", "Is it possible to pause my membership for two months instead of cancelling?"),
    ],
    "General Inquiry": [
        ("Low", "Do you have a student discount?"),
        ("Low", "What are your support hours during public holidays?"),
        ("Low", "I want to give feedback: the new layout is easy to use."),
        ("Medium", "Can someone call me back about enterprise pricing for 200 users?"),
        ("Low", "I would like to partner with your company as a reseller."),
        ("Low", "Where can I find your product documentation and tutorials?"),
        ("Low", "Your website has a spelling mistake on the pricing page."),
        ("Low", "Is the {product} available in Hindi?"),
        ("High", "We have a regulator audit tomorrow and need your security compliance certificate urgently."),
        ("Medium", "I need a quote for 50 licences by the end of the week."),
        ("Low", "Can you explain the difference between the Basic and Pro plans?"),
    ],
}

SLOTS = {
    "plan": ["Basic", "Pro", "Premium", "Starter"],
    "amount": ["Rs 499", "Rs 1,299", "$29.99", "Rs 2,450", "EUR 15"],
    "month": ["January", "March", "June", "September", "November"],
    "order_id": ["ORD-1042", "ORD-3381", "ORD-5710", "ORD-8826"],
    "city": ["Pune", "Ahmedabad", "Chennai", "Jaipur"],
    "product": ["Notes", "Planner", "Reports", "Chat"],
}
GREETINGS = ["", "Hi team,", "Hello,", "Good morning,", "Dear support,"]
ENDINGS = ["", "Thanks.", "Please help.", "Waiting for your reply.", "Regards, Asha"]


def _fill(template: str, rng: random.Random) -> str:
    values = {k: rng.choice(v) for k, v in SLOTS.items()}
    return template.format(**values)


def generate_training_tickets(variants_per_template: int = 6, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame of simulated tickets. Same seed -> same data."""
    rng = random.Random(seed)
    rows = []
    for category, templates in TEMPLATES.items():
        code = "".join(w[0] for w in category.replace("&", "").split()).upper()
        for index, (urgency, template) in enumerate(templates, start=1):
            template_id = f"{code}{index:02d}"
            seen = set()
            attempts = 0
            while len(seen) < variants_per_template and attempts < 100:
                attempts += 1
                body = _fill(template, rng)
                text = " ".join(p for p in [rng.choice(GREETINGS), body, rng.choice(ENDINGS)] if p)
                if rng.random() < 0.15:
                    text = text.lower()          # small noise: some people type in lowercase
                if rng.random() < 0.2:
                    text = text.rstrip(".")       # ...or forget the full stop
                seen.add(text)
            for text in sorted(seen):
                rows.append({"text": text, "category": category, "urgency": urgency, "template_id": template_id})
    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"T{i:04d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    data = generate_training_tickets()
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRAIN_PATH, index=False)
    print(f"Wrote {len(data)} simulated tickets to {TRAIN_PATH}")
    print(data["category"].value_counts().to_string())
