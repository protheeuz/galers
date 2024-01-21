"""Add image_path to DetectionResult

Revision ID: 9b6497ded32f
Revises: c9d576f8fbc8
Create Date: 2024-01-21 17:18:27.825920

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9b6497ded32f'
down_revision = 'c9d576f8fbc8'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('detection_result', schema=None) as batch_op:
        batch_op.add_column(sa.Column('image_path', sa.String(length=255), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('detection_result', schema=None) as batch_op:
        batch_op.drop_column('image_path')

    # ### end Alembic commands ###
