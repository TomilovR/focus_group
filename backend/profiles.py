from faker import Faker
import random
from models import Persona
from database import SessionLocal, PersonaModel, init_db

fake = Faker('ru_RU')

ROLES = [
    ('CTO', 'Технический директор'),
    ('Marketing Manager', 'Маркетолог'),
    ('CEO', 'Генеральный директор'),
    ('Sales Director', 'Директор по продажам'),
    ('HR Director', 'HR Директор'),
    ('Developer', 'Разработчик')
]

INDUSTRIES = ['FinTech', 'EdTech', 'E-commerce', 'SaaS', 'AgroTech', 'MedTech']

PSYCHOGRAPHICS = [
    "Прагматик, ценит краткость и цифры.",
    "Визионер, ищет новые возможности для роста.",
    "Скептик, требует доказательств и кейсов.",
    "Новатор, любит тестировать новые инструменты.",
    "Консерватор, предпочитает проверенные решения."
]

def generate_personas(count: int = 5, audience_id: str = None) -> list[Persona]:
    # Ensure DB is initialized (create tables if not exist)
    try:
        init_db()
    except Exception as e:
        print(f"DB Init failed (check connection): {e}")
        # Fallback to in-memory generation if DB fails
        return _generate_random_personas(count)

    db = SessionLocal()
    
    try:
        query = db.query(PersonaModel)
        
        if audience_id:
            try:
                aud_id_int = int(audience_id)
                query = query.filter(PersonaModel.audience_id == aud_id_int)
                print(f"Filtering by audience_id: {aud_id_int}")
            except ValueError:
                print(f"Invalid audience_id: {audience_id}")
        
        # Get all matching personas first to see how many we have
        available_personas = query.all()
        
        if not available_personas:
             print(f"No personas found for audience {audience_id}. Generating random ones.")
             return _generate_random_personas(count)

        # If we need more than available, we might need to duplicate or just return what we have
        # For now, let's just return a random sample of the requested size from the available ones
        # If requested count is larger than available, return all available (or loop? let's return all available for now)
        
        import random
        selected_models = []
        if len(available_personas) <= count:
            selected_models = available_personas
        else:
            selected_models = random.sample(available_personas, count)
            
        print(f"Loaded {len(selected_models)} personas from DB (requested {count}).")
        
        return [Persona(
            id=p.id,
            name=p.name,
            role=p.role,
            company=p.company,
            avatar=p.avatar,
            psychographics=p.psychographics,
            pastBehavior=p.past_behavior
        ) for p in selected_models]
    except Exception as e:
        print(f"DB Error: {e}")
        return _generate_random_personas(count)
    finally:
        db.close()

def _generate_random_personas(count: int) -> list[Persona]:
    print("Fallback: Generating random personas (DB unavailable)")
    personas = []
    for i in range(count):
        role_en, role_ru = random.choice(ROLES)
        industry = random.choice(INDUSTRIES)
        
        persona = Persona(
            id=str(i + 1),
            name=fake.name(),
            role=role_en,
            company=f"{fake.company()} ({industry})",
            avatar=random.choice(['👨‍💻', '👩‍💼', '🤵', '👷', '👩‍🎨', '🦸‍♂️']),
            psychographics=random.choice(PSYCHOGRAPHICS),
            pastBehavior=f"Часто открывает письма про {industry}, но редко отвечает."
        )
        personas.append(persona)
    return personas
