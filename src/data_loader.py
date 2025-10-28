"""
Sample Data Generator for Support Tickets

Generates realistic support ticket data for testing and demonstration.
Creates a diverse dataset with various categories, priorities, and resolutions.

Categories covered:
- Authentication & Login
- Performance & Speed
- Payment & Billing
- Features & Functionality
- Bugs & Errors
- Account Management
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TicketDataGenerator:
    """Generate realistic support ticket data."""
    
    # Ticket templates by category
    TEMPLATES = {
        'Authentication': [
            {
                'title': 'Cannot log into my account',
                'description': 'I have been trying to log into my account for the past hour but keep getting an "invalid credentials" error. I am sure my password is correct.',
                'resolution': 'Reset password and cleared browser cache. Issue was caused by expired session tokens. Advised customer to use incognito mode if issue persists.',
                'priority': 'high',
                'tags': ['login', 'authentication', 'credentials'],
                'resolution_time': 4.5
            },
            {
                'title': 'Password reset link not working',
                'description': 'I requested a password reset but the link in the email does not work. When I click it, I get a 404 error page.',
                'resolution': 'Link had expired. Sent new reset link with extended validity. Updated email template to include expiration notice.',
                'priority': 'high',
                'tags': ['password', 'reset', 'email'],
                'resolution_time': 2.0
            },
            {
                'title': 'Two-factor authentication issues',
                'description': 'The 2FA code is not being sent to my phone. I need to access my account urgently but cannot complete the login process.',
                'resolution': 'Phone number on file was outdated. Updated contact info and temporarily disabled 2FA for account recovery. Re-enabled after verification.',
                'priority': 'high',
                'tags': ['2fa', 'authentication', 'security'],
                'resolution_time': 6.0
            },
            {
                'title': 'Account locked after failed login attempts',
                'description': 'My account is locked after I entered the wrong password a few times. How can I unlock it?',
                'resolution': 'Unlocked account manually after identity verification. Recommended using password manager to prevent future lockouts.',
                'priority': 'medium',
                'tags': ['locked', 'login', 'security'],
                'resolution_time': 1.5
            },
            {
                'title': 'SSO login not redirecting properly',
                'description': 'When I try to sign in using my company SSO, the page redirects but then shows a blank screen.',
                'resolution': 'Cookie permission issue in browser. Instructed user to enable third-party cookies for SSO domain. Issue resolved.',
                'priority': 'medium',
                'tags': ['sso', 'redirect', 'login'],
                'resolution_time': 3.0
            }
        ],
        'Performance': [
            {
                'title': 'Application is very slow',
                'description': 'The app takes forever to load. Every action I perform has a 5-10 second delay. This started happening yesterday.',
                'resolution': 'Cleared application cache and optimized database queries. Identified N+1 query issue in recent deployment. Rolled back problematic code.',
                'priority': 'high',
                'tags': ['performance', 'slow', 'latency'],
                'resolution_time': 8.0
            },
            {
                'title': 'App crashes on startup',
                'description': 'Every time I try to launch the application, it crashes immediately with no error message. I have tried restarting my computer.',
                'resolution': 'Corrupted local cache file. Cleared app data and reinstalled. Issue related to recent update conflicting with existing settings.',
                'priority': 'critical',
                'tags': ['crash', 'startup', 'error'],
                'resolution_time': 5.0
            },
            {
                'title': 'Dashboard not loading',
                'description': 'When I navigate to the main dashboard, the page stays blank and nothing loads. Other pages work fine.',
                'resolution': 'API endpoint timeout due to large dataset. Implemented pagination and caching. Dashboard now loads in under 2 seconds.',
                'priority': 'high',
                'tags': ['dashboard', 'loading', 'performance'],
                'resolution_time': 10.0
            },
            {
                'title': 'File upload extremely slow',
                'description': 'Uploading a 10MB file takes over 10 minutes. This used to take only seconds. Is there an issue with the servers?',
                'resolution': 'Network throttling on user side and CDN misconfiguration. Switched to different CDN region and advised user to check internet connection.',
                'priority': 'medium',
                'tags': ['upload', 'performance', 'network'],
                'resolution_time': 4.0
            },
            {
                'title': 'Memory usage keeps increasing',
                'description': 'The app uses more and more RAM the longer I keep it open. After a few hours it uses 4GB and my computer slows down.',
                'resolution': 'Memory leak in event listener cleanup. Deployed hotfix that properly disposes of listeners. Memory usage now stable.',
                'priority': 'high',
                'tags': ['memory', 'performance', 'leak'],
                'resolution_time': 12.0
            }
        ],
        'Payment': [
            {
                'title': 'Payment failed during checkout',
                'description': 'My credit card was declined during checkout but my bank shows the charge went through. What happened to my order?',
                'resolution': 'Payment gateway timeout caused duplicate charge. Refunded duplicate transaction. Order processed successfully. Updated timeout settings.',
                'priority': 'high',
                'tags': ['payment', 'checkout', 'refund'],
                'resolution_time': 6.0
            },
            {
                'title': 'Unable to update payment method',
                'description': 'I am trying to update my credit card information but the form does not accept my new card. It says "invalid card number".',
                'resolution': 'Card validation regex was too strict for certain card types. Updated validation rules to accept all major card issuers.',
                'priority': 'medium',
                'tags': ['payment', 'card', 'validation'],
                'resolution_time': 3.0
            },
            {
                'title': 'Subscription not cancelled',
                'description': 'I cancelled my subscription last week but was still charged today. Can I get a refund?',
                'resolution': 'Cancellation did not process due to billing cycle timing. Issued refund and confirmed cancellation. Added confirmation email for cancellations.',
                'priority': 'high',
                'tags': ['subscription', 'billing', 'cancellation'],
                'resolution_time': 2.5
            },
            {
                'title': 'Invoice showing wrong amount',
                'description': 'My invoice shows $150 but I was only supposed to be charged $99 according to my plan. Please explain the charges.',
                'resolution': 'Additional charges for overage usage. Sent detailed breakdown of usage fees. Customer understood and accepted charges.',
                'priority': 'medium',
                'tags': ['invoice', 'billing', 'charges'],
                'resolution_time': 1.5
            },
            {
                'title': 'Discount code not applying',
                'description': 'I have a 20% discount code but when I enter it at checkout, nothing happens. The code is SAVE20.',
                'resolution': 'Code expired yesterday. Issued new code with extended validity as courtesy. Updated code expiration notifications.',
                'priority': 'low',
                'tags': ['discount', 'promo', 'checkout'],
                'resolution_time': 2.0
            }
        ],
        'Features': [
            {
                'title': 'Cannot find export feature',
                'description': 'I need to export my data to CSV but cannot find the export button anywhere. Where is this feature located?',
                'resolution': 'Export feature is in Settings > Data > Export. Sent video tutorial. Improved UI to make export more discoverable.',
                'priority': 'low',
                'tags': ['export', 'feature', 'ui'],
                'resolution_time': 1.0
            },
            {
                'title': 'Dark mode not working',
                'description': 'I enabled dark mode in settings but the app is still showing in light mode. I have refreshed the page multiple times.',
                'resolution': 'Browser cache issue. Cleared cache and reset theme preference. Dark mode now working correctly.',
                'priority': 'low',
                'tags': ['dark-mode', 'theme', 'settings'],
                'resolution_time': 0.5
            },
            {
                'title': 'Notifications not arriving',
                'description': 'I am not receiving any email notifications even though they are enabled in my settings. I have checked my spam folder.',
                'resolution': 'Email address had typo. Corrected email and resent verification. Added email verification step during signup.',
                'priority': 'medium',
                'tags': ['notifications', 'email', 'settings'],
                'resolution_time': 2.0
            },
            {
                'title': 'Request for bulk editing feature',
                'description': 'It would be really helpful to have a bulk edit feature so I can update multiple items at once instead of one by one.',
                'resolution': 'Feature request logged and prioritized for Q2 roadmap. Provided workaround using API for power users.',
                'priority': 'low',
                'tags': ['feature-request', 'bulk-edit', 'enhancement'],
                'resolution_time': 1.0
            },
            {
                'title': 'Mobile app missing features',
                'description': 'The mobile app is missing several features that are available on the web version. Will these be added?',
                'resolution': 'Mobile parity is in development. Shared roadmap timeline. Key features will be added in next major release.',
                'priority': 'medium',
                'tags': ['mobile', 'features', 'parity'],
                'resolution_time': 1.5
            }
        ],
        'Bugs': [
            {
                'title': 'Data not saving properly',
                'description': 'I make changes to my profile but when I come back later, none of the changes are saved. Very frustrating!',
                'resolution': 'Race condition in save API. Deployed fix that properly queues save requests. Data now persists correctly.',
                'priority': 'critical',
                'tags': ['bug', 'save', 'data-loss'],
                'resolution_time': 8.0
            },
            {
                'title': 'Images not displaying',
                'description': 'All images on the website are showing as broken links. This is happening on both Chrome and Firefox.',
                'resolution': 'CDN outage. Switched to backup CDN provider. Images restored within 30 minutes. Implemented automatic CDN failover.',
                'priority': 'critical',
                'tags': ['images', 'cdn', 'outage'],
                'resolution_time': 1.0
            },
            {
                'title': 'Search results showing duplicates',
                'description': 'When I search for anything, I see the same results repeated multiple times in the list.',
                'resolution': 'Elasticsearch indexing issue. Rebuilt search index and removed duplicates. Added deduplication logic to prevent recurrence.',
                'priority': 'medium',
                'tags': ['search', 'duplicates', 'bug'],
                'resolution_time': 4.0
            },
            {
                'title': 'Calendar events showing wrong timezone',
                'description': 'All my calendar events are showing times that are 3 hours off. My timezone is set correctly in my profile.',
                'resolution': 'Timezone conversion bug in calendar display. Fixed DST handling logic. Events now display in correct local time.',
                'priority': 'medium',
                'tags': ['calendar', 'timezone', 'bug'],
                'resolution_time': 6.0
            },
            {
                'title': 'Print layout completely broken',
                'description': 'When I try to print reports, the formatting is completely wrong. Text is cut off and images are overlapping.',
                'resolution': 'CSS print stylesheet was missing. Added proper print styles and tested across browsers. Print layout now clean.',
                'priority': 'low',
                'tags': ['print', 'layout', 'css'],
                'resolution_time': 3.0
            }
        ],
        'Account': [
            {
                'title': 'Want to delete my account',
                'description': 'I would like to permanently delete my account and all associated data. How do I do this?',
                'resolution': 'Processed account deletion request per GDPR requirements. Confirmed data removal and sent confirmation email.',
                'priority': 'high',
                'tags': ['account', 'deletion', 'gdpr'],
                'resolution_time': 2.0
            },
            {
                'title': 'Cannot change email address',
                'description': 'I am trying to update my email address but it says the new email is already in use. It is my email!',
                'resolution': 'Email was associated with inactive account. Merged accounts and updated primary email. Sent verification to new address.',
                'priority': 'medium',
                'tags': ['email', 'account', 'update'],
                'resolution_time': 3.0
            },
            {
                'title': 'Need to transfer account ownership',
                'description': 'I am leaving the company and need to transfer my account to my colleague. How can this be done?',
                'resolution': 'Initiated ownership transfer process. Verified both parties and transferred all data and settings. Updated billing information.',
                'priority': 'high',
                'tags': ['transfer', 'ownership', 'account'],
                'resolution_time': 5.0
            },
            {
                'title': 'Upgrade to premium not reflecting',
                'description': 'I upgraded to the premium plan yesterday but my account still shows as basic. When will this update?',
                'resolution': 'Payment processing delay. Manually upgraded account and applied backdated premium features. Issue with payment processor integration fixed.',
                'priority': 'high',
                'tags': ['upgrade', 'premium', 'billing'],
                'resolution_time': 2.0
            },
            {
                'title': 'Profile picture upload failing',
                'description': 'I cannot upload a new profile picture. It says "file too large" but my image is only 1MB.',
                'resolution': 'File type validation was rejecting HEIC format. Added HEIC support and increased size limit to 5MB. Upload successful.',
                'priority': 'low',
                'tags': ['profile', 'upload', 'image'],
                'resolution_time': 2.5
            }
        ]
    }
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.customer_tiers = ['basic', 'premium', 'enterprise']
        self.source_channels = ['email', 'chat', 'phone', 'portal']
        self.statuses = ['resolved', 'resolved', 'resolved', 'open', 'in_progress']  # Bias toward resolved
        
    def generate_tickets(self, count: int = 500) -> pd.DataFrame:
        """
        Generate a dataset of realistic support tickets.
        
        Args:
            count: Number of tickets to generate
            
        Returns:
            DataFrame with ticket data
        """
        logger.info(f"Generating {count} support tickets...")
        
        tickets = []
        categories = list(self.TEMPLATES.keys())
        
        for i in range(count):
            # Pick random category
            category = random.choice(categories)
            
            # Pick random template from category
            template = random.choice(self.TEMPLATES[category])
            
            # Add some variation to make tickets more unique
            title = template['title']
            description = template['description']
            
            # Randomly add status
            status = random.choice(self.statuses)
            
            # Create timestamp (random date in last 6 months)
            days_ago = random.randint(0, 180)
            created_at = datetime.now() - timedelta(days=days_ago)
            
            # Resolution data (only for resolved tickets)
            resolution = None
            resolved_at = None
            resolution_time = None
            satisfaction_score = None
            
            if status == 'resolved':
                resolution = template['resolution']
                resolution_time = template['resolution_time'] + random.uniform(-1, 2)
                resolved_at = created_at + timedelta(hours=resolution_time)
                satisfaction_score = random.choices([3, 4, 5], weights=[0.1, 0.3, 0.6])[0]
            
            ticket = {
                'ticket_id': f'TKT-{10000 + i}',
                'created_at': created_at,
                'updated_at': resolved_at or created_at,
                'title': title,
                'description': description,
                'category': category,
                'priority': template['priority'],
                'status': status,
                'resolution': resolution,
                'resolved_at': resolved_at,
                'resolution_time_hours': resolution_time,
                'customer_id': f'CUST-{random.randint(1000, 9999)}',
                'customer_tier': random.choice(self.customer_tiers),
                'assigned_agent_id': f'AGENT-{random.randint(100, 199)}' if status != 'open' else None,
                'assigned_team': f'{category.lower()}_team',
                'satisfaction_score': satisfaction_score,
                'reopened_count': 0 if random.random() > 0.1 else 1,
                'combined_text': f'{title} {description}',
                'tags': template['tags'],
                'source_channel': random.choice(self.source_channels),
                'language': 'en'
            }
            
            tickets.append(ticket)
        
        df = pd.DataFrame(tickets)
        
        logger.info(f"Generated {len(df)} tickets across {len(categories)} categories")
        logger.info(f"Status distribution: {df['status'].value_counts().to_dict()}")
        
        return df
    
    def get_sample_new_tickets(self) -> List[Dict[str, str]]:
        """
        Get sample new tickets for testing similarity search.
        
        Returns:
            List of new ticket dicts
        """
        return [
            {
                'title': 'Login credentials not working',
                'description': 'I keep getting an error when trying to sign in. It says my password is incorrect but I know it is right.'
            },
            {
                'title': 'App freezing constantly',
                'description': 'The application keeps freezing every few minutes. I have to restart it multiple times per day.'
            },
            {
                'title': 'Charged twice for my subscription',
                'description': 'I see two charges on my credit card for this month. Can someone help me get a refund?'
            },
            {
                'title': 'How do I export my data?',
                'description': 'I need to download all my data but cannot find where to do this. Is there an export option?'
            },
            {
                'title': 'Pictures not loading on the site',
                'description': 'None of the images are showing up. I just see broken image icons everywhere.'
            }
        ]


if __name__ == "__main__":
    # Test data generation
    generator = TicketDataGenerator()
    df = generator.generate_tickets(count=50)
    
    print(f"\n✅ Generated {len(df)} tickets")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nStatus distribution:")
    print(df['status'].value_counts())
    print(f"\nSample ticket:")
    print(df[['ticket_id', 'title', 'category', 'status']].head(3).to_string(index=False))
