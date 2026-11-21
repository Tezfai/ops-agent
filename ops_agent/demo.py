import uuid

DEMO_CUSTOMER_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


def demo_id(name: str) -> uuid.UUID:
    return uuid.uuid5(DEMO_CUSTOMER_ID, name)
