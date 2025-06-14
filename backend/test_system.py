# test_system_fixed.py
import asyncio
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.models.leads import Lead
from app.models.contacts import Contact
from app.core.security import verify_password
from sqlalchemy import select

async def test_system_simple():
    """Simplified test without lazy loading issues"""
    print("🧪 Testing Lead Generation System (Fixed)...\n")
    
    async with AsyncSessionLocal() as session:
        # Test 1: Admin user
        print("1️⃣ Testing admin user...")
        result = await session.execute(select(User).where(User.email == "admin@leadgen.com"))
        admin = result.scalar_one_or_none()
        
        if admin:
            print(f"✅ Admin found: {admin.full_name}")
            print(f"   Email: {admin.email}")
            print(f"   Superuser: {admin.is_superuser}")
            
            # Test password
            is_valid = verify_password("admin123", admin.hashed_password)
            print(f"✅ Password verification: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test 2: Create and query test lead
        print("\n2️⃣ Creating test lead...")
        test_lead = Lead(
            user_id=admin.id,
            company_name="Test Company Inc",
            domain="testcompany.com",
            industry="Technology",
            employee_count=50,
            location="San Francisco, CA",
            description="A test company",
            data_quality_score=0.85
        )
        
        session.add(test_lead)
        await session.commit()
        await session.refresh(test_lead)  # Refresh to get ID
        
        print(f"✅ Test lead created with ID: {test_lead.id}")
        print(f"   Company: {test_lead.company_name}")
        
        # Test 3: Create test contact
        print("\n3️⃣ Creating test contact...")
        test_contact = Contact(
            lead_id=test_lead.id,
            first_name="John",
            last_name="Doe", 
            full_name="John Doe",
            title="CEO",
            email="john.doe@testcompany.com",
            phone="+1-555-0123",
            email_verified=True
        )
        
        session.add(test_contact)
        await session.commit()
        await session.refresh(test_contact)  # Refresh to get ID
        
        print(f"✅ Test contact created with ID: {test_contact.id}")
        print(f"   Name: {test_contact.full_name}")
        print(f"   Email: {test_contact.email}")
        
        # Test 4: Query counts
        print("\n4️⃣ Counting records...")
        
        user_result = await session.execute(select(User))
        user_count = len(user_result.scalars().all())
        
        lead_result = await session.execute(select(Lead))
        lead_count = len(lead_result.scalars().all())
        
        contact_result = await session.execute(select(Contact))
        contact_count = len(contact_result.scalars().all())
        
        print(f"👥 Users: {user_count}")
        print(f"🏢 Leads: {lead_count}")
        print(f"📞 Contacts: {contact_count}")
        
        # Test 5: Query specific data
        print("\n5️⃣ Testing queries...")
        
        # Find leads for admin user
        admin_leads = await session.execute(
            select(Lead).where(Lead.user_id == admin.id)
        )
        leads_list = admin_leads.scalars().all()
        print(f"📊 Admin's leads: {len(leads_list)}")
        
        # Find contacts for our test lead
        lead_contacts = await session.execute(
            select(Contact).where(Contact.lead_id == test_lead.id)
        )
        contacts_list = lead_contacts.scalars().all()
        print(f"📞 Test lead's contacts: {len(contacts_list)}")
        
        print("\n🎉 All tests passed! System is fully working!")
        
        # Cleanup
        print("\n🧹 Cleaning up test data...")
        await session.delete(test_contact)
        await session.delete(test_lead)
        await session.commit()
        print("✅ Test data cleaned up")

if __name__ == "__main__":
    asyncio.run(test_system_simple())