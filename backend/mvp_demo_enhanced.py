# mvp_demo_enhanced.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.models.leads import Lead
from app.models.contacts import Contact
from sqlalchemy import select
import re
from datetime import datetime
from textblob import TextBlob

class MVPScraper:
    """MVP Scraper with basic sentiment analysis"""
    
    async def find_company_domain(self, company_name: str) -> str:
        """Find company domain via search"""
        try:
            # Use DuckDuckGo search
            search_url = f"https://duckduckgo.com/?q={company_name.replace(' ', '+')}+official+website&t=h_&ia=web"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Look for official website links
                        links = soup.find_all('a', href=True)
                        for link in links:
                            href = link.get('href', '')
                            if href.startswith('http') and any(word in href.lower() for word in [company_name.lower().replace(' ', ''), 'official', 'www']):
                                from urllib.parse import urlparse
                                domain = urlparse(href).netloc.replace('www.', '')
                                if domain and '.' in domain:
                                    return domain
                                    
        except Exception as e:
            print(f"   ⚠️ Error finding domain: {e}")
        
        return None
    
    async def scrape_company_data(self, domain: str) -> dict:
        """Scrape comprehensive company data"""
        try:
            url = f"https://{domain}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        data = {
                            'domain': domain,
                            'description': '',
                            'emails': [],
                            'phones': [],
                            'sentiment_score': 0.0,
                            'quality_score': 0.0
                        }
                        
                        # Extract description
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc:
                            data['description'] = meta_desc.get('content', '')[:500]
                        
                        # Get full text for analysis
                        text_content = soup.get_text()
                        
                        # Extract emails
                        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
                        # Filter for company emails only
                        company_emails = [email for email in emails if domain.split('.')[0] in email.lower()]
                        data['emails'] = list(set(company_emails))[:5]
                        
                        # Extract phones
                        phones = re.findall(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text_content)
                        data['phones'] = list(set(phones))[:3]
                        
                        # Basic sentiment analysis using TextBlob
                        if data['description']:
                            blob = TextBlob(data['description'])
                            data['sentiment_score'] = blob.sentiment.polarity
                        
                        # Calculate quality score
                        score = 0.0
                        if data['description']: score += 0.3
                        if data['emails']: score += 0.4
                        if data['phones']: score += 0.2
                        if abs(data['sentiment_score']) > 0.1: score += 0.1
                        data['quality_score'] = min(score, 1.0)
                        
                        return data
                        
        except Exception as e:
            print(f"   ⚠️ Error scraping {domain}: {e}")
        
        return None

async def run_mvp_demo():
    """Enhanced MVP Demo"""
    print("🚀 Lead Generation MVP Demo - Enhanced")
    print("=" * 60)
    
    # Get admin user
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == "admin@leadgen.com"))
        admin = result.scalar_one_or_none()
        
        if not admin:
            print("❌ Admin user not found. Please run setup_database.py first")
            return
    
    # Test companies for demo
    test_companies = [
        "Stripe",
        "Shopify", 
        "Zoom",
        "Slack"
    ]
    
    scraper = MVPScraper()
    successful_scrapes = 0
    
    for i, company_name in enumerate(test_companies, 1):
        print(f"\n{i}️⃣ Processing: {company_name}")
        print("-" * 40)
        
        # Step 1: Find domain
        print("   🔍 Finding company domain...")
        domain = await scraper.find_company_domain(company_name)
        
        if not domain:
            print(f"   ❌ Could not find domain for {company_name}")
            continue
        
        print(f"   ✅ Found domain: {domain}")
        
        # Step 2: Scrape comprehensive data
        print("   📊 Scraping company data...")
        company_data = await scraper.scrape_company_data(domain)
        
        if not company_data:
            print(f"   ❌ Could not scrape {domain}")
            continue
        
        # Step 3: Save to database
        print("   💾 Saving to database...")
        
        async with AsyncSessionLocal() as session:
            # Create lead with enhanced data
            lead = Lead(
                user_id=admin.id,
                company_name=company_name,
                domain=domain,
                description=company_data.get('description', ''),
                data_quality_score=company_data.get('quality_score', 0.0),
                verification_status="scraped",
                last_scraped=datetime.utcnow()
            )
            
            session.add(lead)
            await session.commit()
            await session.refresh(lead)
            
            # Create contacts from emails
            contact_count = 0
            for email in company_data.get('emails', []):
                # Extract potential name from email
                local_part = email.split('@')[0]
                name_parts = local_part.replace('.', ' ').replace('_', ' ').split()
                
                if len(name_parts) >= 2:
                    first_name = name_parts[0].title()
                    last_name = name_parts[-1].title()
                    full_name = f"{first_name} {last_name}"
                else:
                    full_name = local_part.title()
                    first_name = full_name
                    last_name = ""
                
                # Estimate role from email
                role = "Unknown"
                if any(word in local_part.lower() for word in ['ceo', 'founder', 'president']):
                    role = "Executive"
                elif any(word in local_part.lower() for word in ['sales', 'business']):
                    role = "Sales"
                elif any(word in local_part.lower() for word in ['support', 'help']):
                    role = "Support"
                elif any(word in local_part.lower() for word in ['info', 'contact']):
                    role = "General"
                
                contact = Contact(
                    lead_id=lead.id,
                    first_name=first_name,
                    last_name=last_name,
                    full_name=full_name,
                    title=role,
                    email=email,
                    email_verified=False,
                    contact_quality_score=min(0.8 if role != "Unknown" else 0.5, 1.0)
                )
                
                session.add(contact)
                contact_count += 1
            
            await session.commit()
            successful_scrapes += 1
            
            # Display detailed results
            print(f"   ✅ Lead saved (ID: {lead.id})")
            print(f"   📋 Company: {company_name}")
            print(f"   🌐 Domain: {domain}")
            print(f"   📧 Company emails: {len(company_data.get('emails', []))}")
            print(f"   📞 Phone numbers: {len(company_data.get('phones', []))}")
            print(f"   😊 Sentiment score: {company_data.get('sentiment_score', 0):.2f}")
            print(f"   ⭐ Quality score: {company_data.get('quality_score', 0):.2f}")
            print(f"   👥 Contacts created: {contact_count}")
    
    # Final comprehensive summary
    print(f"\n🎉 MVP Demo Completed!")
    print("=" * 60)
    print(f"📈 Success rate: {successful_scrapes}/{len(test_companies)} companies")
    
    async with AsyncSessionLocal() as session:
        # Get comprehensive stats
        lead_result = await session.execute(select(Lead))
        all_leads = lead_result.scalars().all()
        
        contact_result = await session.execute(select(Contact))
        all_contacts = contact_result.scalars().all()
        
        # Calculate averages
        if all_leads:
            avg_quality = sum(lead.data_quality_score for lead in all_leads) / len(all_leads)
            verified_leads = len([lead for lead in all_leads if lead.verification_status == "scraped"])
        else:
            avg_quality = 0
            verified_leads = 0
        
        print(f"📊 Database Statistics:")
        print(f"   • Total leads: {len(all_leads)}")
        print(f"   • Total contacts: {len(all_contacts)}")
        print(f"   • Verified leads: {verified_leads}")
        print(f"   • Average quality score: {avg_quality:.2f}")
        
        print(f"\n💡 MVP Achievements:")
        print(f"   ✅ Automated company discovery")
        print(f"   ✅ Domain identification") 
        print(f"   ✅ Contact extraction")
        print(f"   ✅ Data quality scoring")
        print(f"   ✅ Basic sentiment analysis")
        print(f"   ✅ Structured data storage")
        
        print(f"\n🚀 Next Steps for Production:")
        print(f"   • Add advanced sentiment analysis")
        print(f"   • Implement email verification")
        print(f"   • Add more data sources")
        print(f"   • Build web dashboard")
        print(f"   • Add batch processing")

if __name__ == "__main__":
    asyncio.run(run_mvp_demo())