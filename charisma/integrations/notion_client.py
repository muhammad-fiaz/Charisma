"""Notion API client for fetching user notes and memories with OAuth support"""

import secrets
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

try:
    from notion_client import Client
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()

# Notion OAuth Configuration
NOTION_AUTH_URL = "https://api.notion.com/v1/oauth/authorize"
NOTION_TOKEN_URL = "https://api.notion.com/v1/oauth/token"
REDIRECT_URI = "http://localhost:8888/callback"

# OAuth credentials - User needs to create a Notion integration
# at https://www.notion.so/my-integrations
NOTION_CLIENT_ID = ""  # Will be loaded from config or env
NOTION_CLIENT_SECRET = ""  # Will be loaded from config or env


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""
    
    access_token = None
    refresh_token = None
    bot_id = None
    workspace_id = None
    workspace_name = None
    workspace_icon = None
    duplicated_template_id = None
    owner = None
    error = None
    
    def do_GET(self):
        """Handle OAuth callback GET request"""
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        if "code" in query_params:
            # Success - got authorization code
            auth_code = query_params["code"][0]
            
            # Exchange code for access token
            try:
                import requests
                
                auth_str = f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}"
                import base64
                b64_auth = base64.b64encode(auth_str.encode()).decode()
                
                payload = {
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": REDIRECT_URI,
                }
                
                response = requests.post(
                    NOTION_TOKEN_URL,
                    headers={
                        "Authorization": f"Basic {b64_auth}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    
                    # Store all OAuth response data as per Notion API spec
                    OAuthCallbackHandler.access_token = token_data.get("access_token")
                    OAuthCallbackHandler.refresh_token = token_data.get("refresh_token")
                    OAuthCallbackHandler.bot_id = token_data.get("bot_id")
                    OAuthCallbackHandler.workspace_id = token_data.get("workspace_id")
                    OAuthCallbackHandler.workspace_name = token_data.get("workspace_name")
                    OAuthCallbackHandler.workspace_icon = token_data.get("workspace_icon")
                    OAuthCallbackHandler.duplicated_template_id = token_data.get("duplicated_template_id")
                    OAuthCallbackHandler.owner = token_data.get("owner")
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    success_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Successfully Connected</title>
                        <style>
                            body {{
                                font-family: Arial, sans-serif;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                height: 100vh;
                                margin: 0;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            }}
                            .container {{
                                background: white;
                                padding: 40px;
                                border-radius: 10px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                text-align: center;
                                max-width: 500px;
                            }}
                            .checkmark {{
                                color: #10b981;
                                font-size: 48px;
                                margin-bottom: 20px;
                            }}
                            h1 {{
                                color: #1f2937;
                                margin-bottom: 10px;
                            }}
                            p {{
                                color: #6b7280;
                                margin-bottom: 20px;
                            }}
                            .workspace-info {{
                                background: #f3f4f6;
                                border-radius: 5px;
                                padding: 15px;
                                margin: 20px 0;
                            }}
                            .workspace-name {{
                                font-weight: bold;
                                color: #374151;
                                font-size: 18px;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="checkmark">✅</div>
                            <h1>Successfully Connected!</h1>
                            <p>Your Notion account has been connected to Charisma.</p>
                            <div class="workspace-info">
                                <div class="workspace-name">{OAuthCallbackHandler.workspace_name or 'Your Workspace'}</div>
                            </div>
                            <p>You can close this window and return to the application.</p>
                        </div>
                    </body>
                    </html>
                    """
                    self.wfile.write(success_html.encode())
                    logger.success(f"Notion OAuth successful - Connected to workspace: {OAuthCallbackHandler.workspace_name}")
                else:
                    raise Exception(f"Token exchange failed: {response.text}")
                    
            except Exception as e:
                OAuthCallbackHandler.error = str(e)
                logger.error(f"OAuth token exchange failed: {e}")
                self.send_error_page(str(e))
        
        elif "error" in query_params:
            # OAuth error
            error = query_params.get("error", ["Unknown error"])[0]
            error_desc = query_params.get("error_description", [""])[0]
            OAuthCallbackHandler.error = f"{error}: {error_desc}"
            logger.fail(f"Notion OAuth failed: {error} - {error_desc}")
            self.send_error_page(f"{error}: {error_desc}")
        else:
            self.send_error_page("Invalid callback parameters")
    
    def send_error_page(self, error_msg):
        """Send error HTML page"""
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connection Failed</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 500px;
                }}
                .error {{
                    color: #ef4444;
                    font-size: 48px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: #1f2937;
                    margin-bottom: 10px;
                }}
                p {{
                    color: #6b7280;
                    margin-bottom: 20px;
                }}
                .error-details {{
                    background: #fef2f2;
                    border: 1px solid #fee2e2;
                    padding: 15px;
                    border-radius: 5px;
                    color: #991b1b;
                    font-size: 14px;
                    word-wrap: break-word;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error">❌</div>
                <h1>Connection Failed</h1>
                <p>There was a problem connecting your Notion account.</p>
                <div class="error-details">{error_msg}</div>
                <p style="margin-top: 20px;">You can close this window and try again.</p>
            </div>
        </body>
        </html>
        """
        self.wfile.write(error_html.encode())
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging"""
        pass


class NotionOAuthManager:
    """Manager for Notion OAuth flow"""
    
    def __init__(self, client_id: str = "", client_secret: str = ""):
        """
        Initialize OAuth manager
        
        Args:
            client_id: Notion OAuth client ID
            client_secret: Notion OAuth client secret
        """
        global NOTION_CLIENT_ID, NOTION_CLIENT_SECRET
        NOTION_CLIENT_ID = client_id
        NOTION_CLIENT_SECRET = client_secret
        self.state = secrets.token_urlsafe(32)
        self.server = None
        self.server_thread = None
    
    def get_authorization_url(self) -> str:
        """Generate Notion OAuth authorization URL"""
        params = {
            "client_id": NOTION_CLIENT_ID,
            "response_type": "code",
            "owner": "user",
            "redirect_uri": REDIRECT_URI,
            "state": self.state,
        }
        return f"{NOTION_AUTH_URL}?{urlencode(params)}"
    
    def start_callback_server(self):
        """Start local server to handle OAuth callback"""
        try:
            self.server = HTTPServer(("localhost", 8888), OAuthCallbackHandler)
            self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            logger.info("OAuth callback server started on http://localhost:8888")
        except Exception as e:
            logger.error(f"Failed to start callback server: {e}")
            raise
    
    def stop_callback_server(self):
        """Stop the OAuth callback server"""
        if self.server:
            self.server.shutdown()
            self.server = None
            logger.info("OAuth callback server stopped")
    
    def open_browser(self) -> bool:
        """Open browser for OAuth authorization"""
        try:
            auth_url = self.get_authorization_url()
            logger.info(f"Opening browser for Notion OAuth: {auth_url}")
            webbrowser.open(auth_url)
            return True
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            return False
    
    def wait_for_callback(self, timeout: int = 300) -> Optional[Dict]:
        """
        Wait for OAuth callback and return OAuth data
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            
        Returns:
            Dict with OAuth data if successful, None otherwise
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if OAuthCallbackHandler.access_token:
                # Collect all OAuth data
                oauth_data = {
                    "access_token": OAuthCallbackHandler.access_token,
                    "refresh_token": OAuthCallbackHandler.refresh_token,
                    "bot_id": OAuthCallbackHandler.bot_id,
                    "workspace_id": OAuthCallbackHandler.workspace_id,
                    "workspace_name": OAuthCallbackHandler.workspace_name,
                    "workspace_icon": OAuthCallbackHandler.workspace_icon,
                    "duplicated_template_id": OAuthCallbackHandler.duplicated_template_id,
                    "owner": OAuthCallbackHandler.owner,
                }
                
                # Reset for next auth
                OAuthCallbackHandler.access_token = None
                OAuthCallbackHandler.refresh_token = None
                OAuthCallbackHandler.bot_id = None
                OAuthCallbackHandler.workspace_id = None
                OAuthCallbackHandler.workspace_name = None
                OAuthCallbackHandler.workspace_icon = None
                OAuthCallbackHandler.duplicated_template_id = None
                OAuthCallbackHandler.owner = None
                OAuthCallbackHandler.error = None
                return oauth_data
            
            if OAuthCallbackHandler.error:
                error = OAuthCallbackHandler.error
                # Reset for next auth
                OAuthCallbackHandler.access_token = None
                OAuthCallbackHandler.error = None
                raise Exception(f"OAuth failed: {error}")
            
            time.sleep(0.5)
        
        raise TimeoutError("OAuth callback timeout - user did not complete authorization")
    
    def authenticate(self) -> Optional[Dict]:
        """
        Complete OAuth flow and return OAuth data
        
        Returns:
            Dict with OAuth data if successful, None otherwise
        """
        try:
            # Start callback server
            self.start_callback_server()
            
            # Open browser for authorization
            if not self.open_browser():
                self.stop_callback_server()
                return None
            
            logger.info("Waiting for user to authorize in browser...")
            
            # Wait for callback
            oauth_data = self.wait_for_callback()
            
            # Stop server
            self.stop_callback_server()
            
            return oauth_data
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            self.stop_callback_server()
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: The refresh token from previous OAuth
            
        Returns:
            Dict with new OAuth data if successful, None otherwise
        """
        try:
            import requests
            import base64
            
            auth_str = f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}"
            b64_auth = base64.b64encode(auth_str.encode()).decode()
            
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            
            response = requests.post(
                NOTION_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {b64_auth}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            
            if response.status_code == 200:
                token_data = response.json()
                oauth_data = {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "bot_id": token_data.get("bot_id"),
                    "workspace_id": token_data.get("workspace_id"),
                    "workspace_name": token_data.get("workspace_name"),
                    "workspace_icon": token_data.get("workspace_icon"),
                    "duplicated_template_id": token_data.get("duplicated_template_id"),
                    "owner": token_data.get("owner"),
                }
                logger.success("Access token refreshed successfully")
                return oauth_data
            else:
                logger.error(f"Token refresh failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token refresh exception: {e}")
            return None
            
            # Open browser for authorization
            if not self.open_browser():
                self.stop_callback_server()
                return None
            
            logger.info("Waiting for user to authorize in browser...")
            
            # Wait for callback
            token = self.wait_for_callback()
            
            # Stop server
            self.stop_callback_server()
            
            return token
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            self.stop_callback_server()
            return None


class NotionClient:
    """Client for interacting with Notion API"""

    def __init__(self, api_key: str = "", client_id: str = "", client_secret: str = ""):
        """
        Initialize Notion client
        
        Args:
            api_key: Notion integration token (internal integrations)
            client_id: OAuth client ID (public integrations)
            client_secret: OAuth client secret (public integrations)
        """
        self.api_key = api_key
        self.client_id = client_id
        self.client_secret = client_secret
        self.client = None
        self.oauth_manager = None
        
        # OAuth data storage
        self.access_token = None
        self.refresh_token = None
        self.bot_id = None
        self.workspace_id = None
        self.workspace_name = None
        self.workspace_icon = None
        self.duplicated_template_id = None
        self.owner = None
        
        if api_key:
            try:
                self.client = Client(auth=api_key)
                logger.success("Notion client initialized with API key")
            except Exception as e:
                logger.error(f"Failed to initialize Notion client: {e}")
        elif client_id and client_secret:
            # OAuth mode
            self.oauth_manager = NotionOAuthManager(client_id, client_secret)
            logger.info("Notion client ready for OAuth authentication")
    
    def authenticate_with_browser(self) -> bool:
        """
        Authenticate using browser OAuth flow
        
        Returns:
            True if successful, False otherwise
        """
        if not self.oauth_manager:
            if not self.client_id or not self.client_secret:
                logger.error("OAuth credentials not configured")
                return False
            self.oauth_manager = NotionOAuthManager(self.client_id, self.client_secret)
        
        try:
            logger.info("Starting browser-based Notion authentication...")
            oauth_data = self.oauth_manager.authenticate()
            
            if oauth_data and oauth_data.get("access_token"):
                # Store all OAuth data
                self.access_token = oauth_data.get("access_token")
                self.refresh_token = oauth_data.get("refresh_token")
                self.bot_id = oauth_data.get("bot_id")
                self.workspace_id = oauth_data.get("workspace_id")
                self.workspace_name = oauth_data.get("workspace_name")
                self.workspace_icon = oauth_data.get("workspace_icon")
                self.duplicated_template_id = oauth_data.get("duplicated_template_id")
                self.owner = oauth_data.get("owner")
                
                # Initialize client with OAuth token
                self.client = Client(auth=self.access_token)
                self.api_key = self.access_token
                logger.success(f"Notion authentication successful! Connected to: {self.workspace_name}")
                return True
            else:
                logger.fail("Failed to get access token from Notion")
                return False
                
        except Exception as e:
            logger.error(f"Browser authentication failed: {e}")
            return False
    
    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token
        
        Returns:
            True if successful, False otherwise
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        if not self.oauth_manager:
            if not self.client_id or not self.client_secret:
                logger.error("OAuth credentials not configured")
                return False
            self.oauth_manager = NotionOAuthManager(self.client_id, self.client_secret)
        
        try:
            oauth_data = self.oauth_manager.refresh_token(self.refresh_token)
            
            if oauth_data and oauth_data.get("access_token"):
                # Update all OAuth data
                self.access_token = oauth_data.get("access_token")
                self.refresh_token = oauth_data.get("refresh_token")
                self.bot_id = oauth_data.get("bot_id")
                self.workspace_id = oauth_data.get("workspace_id")
                self.workspace_name = oauth_data.get("workspace_name")
                self.workspace_icon = oauth_data.get("workspace_icon")
                self.duplicated_template_id = oauth_data.get("duplicated_template_id")
                self.owner = oauth_data.get("owner")
                
                # Re-initialize client with new token
                self.client = Client(auth=self.access_token)
                self.api_key = self.access_token
                logger.success("Access token refreshed successfully")
                return True
            else:
                logger.error("Failed to refresh access token")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_oauth_data(self) -> Optional[Dict]:
        """
        Get current OAuth data for storage
        
        Returns:
            Dict with OAuth data or None
        """
        if not self.access_token:
            return None
        
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "bot_id": self.bot_id,
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "workspace_icon": self.workspace_icon,
            "duplicated_template_id": self.duplicated_template_id,
            "owner": self.owner,
        }
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self.client is not None

    def test_connection(self) -> bool:
        """Test Notion API connection"""
        if not self.client:
            return False
        try:
            # Test connection by searching for pages
            self.client.search(
                filter={"property": "object", "value": "page"}, 
                page_size=1
            )
            logger.info("Notion connection test successful")
            return True
        except Exception as e:
            logger.error(f"Notion connection test failed: {e}")
            return False

    def get_all_pages(self, include_content: bool = True) -> List[Dict]:
        """Get all pages accessible to the integration
        
        Args:
            include_content: If True, fetches full page content/blocks for each page
        """
        if not self.client:
            logger.error("Notion client not initialized")
            return []

        all_pages = []
        try:
            has_more = True
            start_cursor = None

            while has_more:
                response = self.client.search(
                    filter={"property": "object", "value": "page"},
                    page_size=100,
                    start_cursor=start_cursor,
                )
                pages = response.get("results", [])
                
                # If include_content is True, fetch full content for each page
                if include_content:
                    logger.info(f"Fetching content for {len(pages)} pages...")
                    for page in pages:
                        page_id = page.get("id")
                        if page_id:
                            # Fetch page content (blocks)
                            content = self.get_page_content(page_id)
                            page["_content"] = content  # Store in custom field
                            
                            # Fetch blocks metadata (full block data)
                            try:
                                blocks_response = self.client.blocks.children.list(block_id=page_id)
                                page["_blocks"] = blocks_response.get("results", [])
                            except Exception as e:
                                logger.warning(f"Could not fetch blocks for page {page_id}: {e}")
                                page["_blocks"] = []
                
                all_pages.extend(pages)
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")

            logger.info(f"Retrieved {len(all_pages)} pages from Notion" + 
                       (" with full content" if include_content else ""))
            return all_pages
        except Exception as e:
            logger.error(f"Error fetching pages from Notion: {e}")
            return []
    
    def get_all_databases(self) -> List[Dict]:
        """Get all databases accessible to the integration
        
        Note: Notion API no longer supports filtering by 'database' in search.
        This method now filters results from the general search.
        """
        if not self.client:
            logger.error("Notion client not initialized")
            return []

        all_databases = []
        try:
            has_more = True
            start_cursor = None

            while has_more:
                # Search without filter, then filter locally
                response = self.client.search(
                    page_size=100,
                    start_cursor=start_cursor,
                )
                # Filter for databases only
                results = response.get("results", [])
                databases = [item for item in results if item.get("object") == "database"]
                all_databases.extend(databases)
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")

            logger.info(f"Retrieved {len(all_databases)} databases from Notion")
            return all_databases
        except Exception as e:
            logger.error(f"Error fetching databases from Notion: {e}")
            return []
    
    def get_all_content(self) -> Dict[str, List[Dict]]:
        """Get both pages and databases"""
        return {
            "pages": self.get_all_pages(),
            "databases": self.get_all_databases()
        }

    def get_page_content(self, page_id: str) -> Optional[str]:
        """Get content from a specific page"""
        if not self.client:
            return None

        try:
            blocks = self.client.blocks.children.list(block_id=page_id)
            content_parts = []

            for block in blocks.get("results", []):
                block_type = block.get("type")
                block_content = block.get(block_type, {})

                if "rich_text" in block_content:
                    text = "".join(
                        [t.get("plain_text", "") for t in block_content["rich_text"]]
                    )
                    content_parts.append(text)

            return "\n".join(content_parts)
        except Exception as e:
            logger.error(f"Error fetching page content: {e}")
            return None

    def format_pages_for_display(self, pages: List[Dict]) -> List[str]:
        """Format pages for display in UI - returns list of display strings"""
        formatted_pages = []

        for page in pages:
            try:
                # Try multiple ways to extract title
                title = None
                
                # Method 1: Check properties for title type
                properties = page.get("properties", {})
                for prop_name, prop_value in properties.items():
                    if prop_value.get("type") == "title":
                        title_parts = prop_value.get("title", [])
                        if title_parts:
                            title = "".join(
                                [t.get("plain_text", "") for t in title_parts]
                            ).strip()
                        break
                
                # Method 2: Check for Name property (common in databases)
                if not title and "Name" in properties:
                    name_prop = properties["Name"]
                    if name_prop.get("type") == "title":
                        title_parts = name_prop.get("title", [])
                        if title_parts:
                            title = "".join(
                                [t.get("plain_text", "") for t in title_parts]
                            ).strip()
                
                # Method 3: Use page ID as fallback
                if not title:
                    page_id = page.get("id", "")
                    if page_id:
                        # Use last 8 characters of ID for readability
                        title = f"Page {page_id[-8:]}"
                    else:
                        title = "Untitled Page"

                # Extract creation date
                created_time = page.get("created_time", "")
                created_date = None
                if created_time:
                    try:
                        created_date = datetime.fromisoformat(
                            created_time.replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                # Format display string with date
                if created_date:
                    date_str = created_date.strftime("%b %d, %Y")
                    display_string = f"{title} ({date_str})"
                else:
                    display_string = title
                
                formatted_pages.append({
                    "display": display_string,
                    "title": title,
                    "date": created_date
                })
                
            except Exception as e:
                logger.error(f"Error formatting page: {e}")
                continue

        # Sort by date (newest first)
        formatted_pages.sort(
            key=lambda x: x.get("date") or datetime.min, reverse=True
        )
        
        # Return only the display strings
        return [page["display"] for page in formatted_pages]
