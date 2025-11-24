from faker import Faker
import random
from backend.database import SessionLocal, init_db, AudienceModel, PersonaModel

fake = Faker('ru_RU')

def populate():
    print("Initializing DB...")
    init_db()
    db = SessionLocal()
    
    # Clear existing data (optional, but good for clean state)
    # db.query(PersonaModel).delete()
    # db.query(AudienceModel).delete()
    # db.commit()
    
    if db.query(AudienceModel).count() > 0:
        print("Audiences already exist. Skipping population.")
        return

    audiences_config = [
        {
            "name": "Технические лидеры (CTO, VP Eng)",
            "type": "B2B",
            "count": 12,
            "roles": [('CTO', 'Технический директор'), ('VP Engineering', 'VP разработки'), ('Tech Lead', 'Тимлид')],
            "industries": ['FinTech', 'SaaS', 'CyberSecurity'],
            "avatars": ['👨‍💻', '🧑‍💻', '👓']
        },
        {
            "name": "HR Директора",
            "type": "B2B",
            "count": 10,
            "roles": [('HR Director', 'HR Директор'), ('Head of Recruitment', 'Руководитель подбора')],
            "industries": ['Retail', 'IT', 'Banking'],
            "avatars": ['👩‍💼', '🧑‍💼', '📋']
        },
        {
            "name": "Маркетологи (SaaS)",
            "type": "B2B",
            "count": 15,
            "roles": [('CMO', 'Директор по маркетингу'), ('Growth Manager', 'Менеджер по росту')],
            "industries": ['SaaS', 'EdTech', 'MarTech'],
            "avatars": ['🚀', '📈', '👩‍🎨']
        },
        {
            "name": "E-commerce Owners",
            "type": "B2C",
            "count": 10,
            "roles": [('Founder', 'Основатель'), ('Owner', 'Владелец')],
            "industries": ['Fashion', 'Electronics', 'Home Decor'],
            "avatars": ['🛍️', '📦', '💎']
        }
    ]

    PSYCHOGRAPHICS = [
        "Прагматик, ценит краткость и цифры.",
        "Визионер, ищет новые возможности для роста.",
        "Скептик, требует доказательств и кейсов.",
        "Новатор, любит тестировать новые инструменты.",
        "Консерватор, предпочитает проверенные решения."
    ]

    print("Generating audiences...")
    
    persona_id_counter = 1
    
    for config in audiences_config:
        audience = AudienceModel(
            name=config["name"],
            type=config["type"],
            description=f"Target audience for {config['name']}"
        )
        db.add(audience)
        db.flush() # Get ID
        
        print(f"  - Creating '{config['name']}' with {config['count']} personas...")
        
        for _ in range(config["count"]):
            role_en, role_ru = random.choice(config["roles"])
            industry = random.choice(config["industries"])
            
            persona = PersonaModel(
                id=str(persona_id_counter),
                audience_id=audience.id,
                name=fake.name(),
                role=role_en,
                company=f"{fake.company()} ({industry})",
                avatar=random.choice(config["avatars"]),
                psychographics=random.choice(PSYCHOGRAPHICS),
                past_behavior=f"Часто открывает письма про {industry}, но редко отвечает."
            )
            db.add(persona)
            persona_id_counter += 1
            
    db.commit()
    print("Done! Database populated.")
    db.close()

if __name__ == "__main__":
    populate()
