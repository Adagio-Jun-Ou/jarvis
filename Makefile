.PHONY: start stop restart status logs ssl health

# Default port for HTTPS
PORT ?= 8443

start:
	@echo "Starting Jarvis on HTTPS port $(PORT)..."
	./venv/bin/python run.py --interface web --port $(PORT)

start-http:
	@echo "Starting Jarvis on HTTP port 8080 (not recommended)..."
	./venv/bin/python run.py --interface web --port 8080

stop:
	@echo "Stopping Jarvis..."
	pkill -f "python run.py" || true

restart: stop
	@sleep 2
	$(MAKE) start

status:
	@curl -sk https://localhost:8443/health 2>/dev/null || echo "Jarvis not running on HTTPS:8443"

logs:
	@tail -f ~/.jarvis/logs/jarvis.log 2>/dev/null || echo "No log file found"

ssl:
	@echo "SSL Certificate Info:"
	@openssl x509 -in ssl/cert.pem -text -noout 2>/dev/null | grep -E "Subject:|DNS:|IP Address:" || echo "Certificate not found"

health:
	@curl -sk https://localhost:8443/health 2>/dev/null && echo "" || echo "Not healthy"

tailscale-status:
	@tailscale status 2>/dev/null || echo "Tailscale not running"

tailscale-url:
	@echo "https://your-host.your-tailnet.ts.net:8443"

# Generate new SSL certificate
ssl-regen:
	@echo "Regenerating SSL certificate..."
	./venv/bin/python -c "
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import datetime
import ipaddress

key = rsa.generate_private_key(65537, 2048, default_backend())
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, 'US'),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, 'California'),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'Jarvis'),
    x509.NameAttribute(NameOID.COMMON_NAME, 'your-host.your-tailnet.ts.net'),
])
cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer) \
    .public_key(key.public_key()).serial_number(x509.random_serial_number()) \
    .not_valid_before(datetime.datetime.now(datetime.timezone.utc)) \
    .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)) \
    .add_extension(x509.SubjectAlternativeName([
        x509.DNSName('localhost'),
        x509.DNSName('your-host'),
        x509.DNSName('your-host.your-tailnet.ts.net'),
        x509.DNSName('*.your-tailnet.ts.net'),
        x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
        x509.IPAddress(ipaddress.IPv4Address('203.0.113.10')),
    ]), False).sign(key, hashes.SHA256(), default_backend())
with open('ssl/key.pem', 'wb') as f:
    f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
with open('ssl/cert.pem', 'wb') as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))
print('Certificate generated successfully')
"

# Install as systemd service
install-systemd:
	@echo "Installing systemd service..."
	sudo cp jarvis.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable jarvis
	@echo "Service installed. Use: sudo systemctl start jarvis"

# Uninstall systemd service
uninstall-systemd:
	@echo "Removing systemd service..."
	sudo systemctl stop jarvis 2>/dev/null || true
	sudo systemctl disable jarvis 2>/dev/null || true
	sudo rm -f /etc/systemd/system/jarvis.service
	sudo systemctl daemon-reload
	@echo "Service removed."
