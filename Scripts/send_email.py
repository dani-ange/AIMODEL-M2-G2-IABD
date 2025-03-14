"""
Script d'envoi automatique des modèles et de la documentation par email.
Utilise les secrets GitHub pour sécuriser les informations de connexion SMTP.
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailSender:
    def __init__(self, smtp_user, smtp_pass, recipients, subject, body):
        """
        Initialise l'EmailSender avec les informations d'authentification et de contenu.

        :param smtp_user: Adresse e-mail de l'expéditeur
        :param smtp_pass: Mot de passe SMTP ou token d'authentification
        :param recipients: Liste des adresses e-mail des destinataires
        :param subject: Sujet de l'email
        :param body: Corps du message
        """
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.recipients = recipients
        self.subject = subject
        self.body = body

    def send_email(self, attachment_path):
        """Envoie l'email avec une pièce jointe."""
        msg = MIMEMultipart()
        msg['From'] = self.smtp_user
        msg['To'] = ', '.join(self.recipients)  # Utilise une chaîne séparée par des virgules
        msg['Subject'] = self.subject

        msg.attach(MIMEText(self.body, 'plain'))

        try:
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                               f'attachment; filename={os.path.basename(attachment_path)}')
                msg.attach(part)
        except FileNotFoundError:
            logging.error(f"Fichier joint introuvable : {attachment_path}")
            return

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
                logging.info(f"Email envoyé avec succès à {', '.join(self.recipients)}!")
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de l'email : {e}")

if __name__ == "__main__":
    recipients = os.getenv('GROUP_EMAILS').split(',') # transforme la chaine de caractere en liste.
    sender = EmailSender(
        smtp_user=os.getenv('SMTP_USER', "ngouedavidrogeryannick@gmail.com"),
        smtp_pass=os.getenv('SMTP_PASS', "Angeline-2007???"),
        recipients="ngouedavidroger@icloud.com",
        subject="Modèle et Documentation",
        body="Veuillez trouver ci-joint le modèle et la documentation générés."
    )
    sender.send_email("output.zip")