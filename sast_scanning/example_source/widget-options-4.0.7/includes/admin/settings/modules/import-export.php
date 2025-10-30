<?php
/**
 * Import & Export Settings Module
 * Settings > Widget Options :: Import & Export
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.4
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Import & Export
 *
 * @since 4.4
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

function widgetopts_settings_import_export(){
    global $widget_options;
    //avoid issue after update
    if( !isset( $widget_options['import_export'] ) ){
        $widget_options['import_export'] = '';
    }
    ?>
    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['import_export'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-import_export" data-module-id="import_export">
		<div class="widgetopts-module-card-content">
			<h2><?php _e( 'Import & Export Widgets', 'widget-options' );?></h2>
			<p class="widgetopts-module-desc">
				<?php _e( 'Import or Export all your widgets with associated sidebar widget area easily.', 'widget-options' );?>
			</p>

			<div class="widgetopts-module-actions hide-if-no-js">
                <?php if( $widget_options['import_export'] == 'activate' ){ ?>
					<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
					<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
				<?php }else{ ?>
					<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
					<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
				<?php } ?>

			</div>

		</div>

		<?php widgetopts_modal_start( $widget_options['import_export'] ); ?>
			<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-download"></span>
			<h3 class="widgetopts-modal-header"><?php _e( 'Import & Export Widgets', 'widget-options' );?></h3>
			<p>
				<?php _e( 'Enabling this feature will give you additional menu under <a href="'. esc_url( admin_url( 'tools.php?page=widgetopts_migrator_settings' ) ) .'"><strong>Tools > Import/Export Widgets</strong></a>. This will give you the easiest option to export and import your widgets. Creating backup and restore of your widgets can be done really simple!', 'widget-options' );?>
			</p>
			<p class="widgetopts-settings-section">
				<?php _e( 'No additional settings available.', 'widget-options' );?>
			</p>
		<?php widgetopts_modal_end( $widget_options['import_export'] ); ?>

	</li>
    <?php
}
add_action( 'widgetopts_module_cards', 'widgetopts_settings_import_export', 64 );
?>
